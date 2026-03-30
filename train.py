"""
train.py
Purpose: Train an EfficientNet-B0 model to classify Alzheimer's disease stages
from MRI images stored in the local data/ directory, while visualizing learning
progress in real time and saving publication-ready figures for documentation.

Research/education only.

Recommended setup before running:
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install -r requirements.txt

Run:
    python train.py
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


DATA_DIR = Path("data")
MODEL_PATH = Path("alz_model.pth")
SCREENSHOTS_DIR = Path("screenshots")
REPORTS_DIR = SCREENSHOTS_DIR / "reports"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0
PREDICTION_MONITOR_INTERVAL = 2
NUM_MONITOR_IMAGES = 4
BATCH_LOSS_SMOOTHING_WINDOW = 20

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# This list documents the expected class labels for the project.
EXPECTED_CLASSES = [
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment",
]


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    true_labels: List[int] = field(default_factory=list)
    predicted_labels: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    sample_images: List[np.ndarray] = field(default_factory=list)
    sample_true_labels: List[int] = field(default_factory=list)
    sample_predicted_labels: List[int] = field(default_factory=list)
    sample_confidences: List[float] = field(default_factory=list)
    sample_tensors: List[torch.Tensor] = field(default_factory=list)
    gradcam_overlays: List[np.ndarray] = field(default_factory=list)
    gradcam_heatmaps: List[np.ndarray] = field(default_factory=list)


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    batch_losses: List[float] = field(default_factory=list)


@dataclass
class VisualizationContext:
    dashboard_figure: plt.Figure
    dashboard_axes: np.ndarray
    confusion_figure: plt.Figure
    confusion_axis: plt.Axes
    sample_figure: plt.Figure
    sample_axes: np.ndarray
    gradcam_figure: plt.Figure
    gradcam_axes: np.ndarray


# This wrapper lets us apply different transforms to train/validation subsets.
class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform: transforms.Compose | None = None) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# This function sets seeds to improve reproducibility across runs.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# This function builds the image transforms used for training and validation.
def get_data_transforms() -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {"train": train_transform, "val": eval_transform}


# This function validates that the dataset folders match the expected labels.
def validate_class_names(found_classes: List[str]) -> None:
    if set(found_classes) != set(EXPECTED_CLASSES):
        raise ValueError(
            "Dataset folders do not match the expected class names.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found: {found_classes}"
        )


# This function loads the dataset, splits it, and returns train/validation loaders.
def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir.resolve()}")

    base_dataset = datasets.ImageFolder(root=data_dir)
    validate_class_names(base_dataset.classes)

    if len(base_dataset) == 0:
        raise ValueError("The dataset is empty. Add MRI images to the data/ directory first.")

    transforms_map = get_data_transforms()
    dataset_size = len(base_dataset)
    val_size = max(1, int(dataset_size * validation_split))
    train_size = dataset_size - val_size

    if train_size <= 0:
        raise ValueError("Dataset is too small to create both training and validation splits.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    train_dataset = TransformedSubset(train_subset, transform=transforms_map["train"])
    val_dataset = TransformedSubset(val_subset, transform=transforms_map["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, base_dataset.classes


# This function creates an EfficientNet-B0 model for the target number of classes.
def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# This function restores normalized image tensors so they can be displayed.
def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


# This function computes a moving average for smoother batch-level loss plots.
def compute_smoothed_losses(loss_values: List[float], window_size: int) -> np.ndarray:
    if not loss_values:
        return np.array([])

    values = np.asarray(loss_values, dtype=np.float32)
    if len(values) < window_size:
        return values

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="valid")


# This function initializes the live matplotlib figures used during training.
def initialize_visualization() -> VisualizationContext:
    plt.ion()
    plt.style.use("seaborn-v0_8-whitegrid")

    dashboard_figure, dashboard_axes = plt.subplots(2, 2, figsize=(18, 12))
    dashboard_figure.suptitle("Alzheimer MRI Training Dashboard", fontsize=18, fontweight="bold")
    dashboard_figure.tight_layout(rect=[0, 0.02, 1, 0.96])

    confusion_figure, confusion_axis = plt.subplots(figsize=(8, 7))
    confusion_figure.suptitle("Validation Confusion Matrix", fontsize=16, fontweight="bold")
    confusion_figure.tight_layout(rect=[0, 0.02, 1, 0.95])

    sample_figure, sample_axes = plt.subplots(2, 2, figsize=(12, 10))
    sample_figure.suptitle("Validation Prediction Monitoring", fontsize=16, fontweight="bold")
    sample_figure.tight_layout(rect=[0, 0.02, 1, 0.95])

    gradcam_figure, gradcam_axes = plt.subplots(NUM_MONITOR_IMAGES, 2, figsize=(12, 4 * NUM_MONITOR_IMAGES))
    gradcam_figure.suptitle("Grad-CAM Attention Monitoring", fontsize=16, fontweight="bold")
    gradcam_figure.tight_layout(rect=[0, 0.02, 1, 0.97])

    return VisualizationContext(
        dashboard_figure=dashboard_figure,
        dashboard_axes=dashboard_axes,
        confusion_figure=confusion_figure,
        confusion_axis=confusion_axis,
        sample_figure=sample_figure,
        sample_axes=sample_axes,
        gradcam_figure=gradcam_figure,
        gradcam_axes=gradcam_axes,
    )


# This function updates the loss and accuracy curves in the dashboard.
def update_training_plot(
    axes: np.ndarray,
    history: TrainingHistory,
    epoch_index: int,
) -> None:
    loss_axis = axes[0, 0]
    accuracy_axis = axes[0, 1]
    batch_axis = axes[1, 1]

    epochs = np.arange(1, len(history.train_losses) + 1)

    loss_axis.clear()
    loss_axis.plot(epochs, history.train_losses, marker="o", linewidth=2.2, label="Train Loss", color="#1d4ed8")
    loss_axis.plot(epochs, history.val_losses, marker="s", linewidth=2.2, label="Val Loss", color="#dc2626")
    loss_axis.set_title(f"Loss Curves Through Epoch {epoch_index}")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend(loc="best")

    accuracy_axis.clear()
    accuracy_axis.plot(
        epochs,
        np.asarray(history.train_accuracies) * 100,
        marker="o",
        linewidth=2.2,
        label="Train Accuracy",
        color="#15803d",
    )
    accuracy_axis.plot(
        epochs,
        np.asarray(history.val_accuracies) * 100,
        marker="s",
        linewidth=2.2,
        label="Val Accuracy",
        color="#7c3aed",
    )
    accuracy_axis.set_title(f"Accuracy Curves Through Epoch {epoch_index}")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.legend(loc="best")

    batch_axis.clear()
    batch_axis.set_title("Batch-Level Training Loss")
    batch_axis.set_xlabel("Batch Step")
    batch_axis.set_ylabel("Loss")
    if history.batch_losses:
        batch_steps = np.arange(1, len(history.batch_losses) + 1)
        batch_axis.plot(batch_steps, history.batch_losses, alpha=0.28, color="#0f172a", label="Batch Loss")

        smoothed_losses = compute_smoothed_losses(history.batch_losses, BATCH_LOSS_SMOOTHING_WINDOW)
        if len(smoothed_losses) > 0:
            if len(history.batch_losses) >= BATCH_LOSS_SMOOTHING_WINDOW:
                smooth_steps = np.arange(BATCH_LOSS_SMOOTHING_WINDOW, len(history.batch_losses) + 1)
            else:
                smooth_steps = batch_steps
            batch_axis.plot(
                smooth_steps,
                smoothed_losses,
                linewidth=2.4,
                color="#ea580c",
                label=f"Smoothed ({BATCH_LOSS_SMOOTHING_WINDOW}-batch)",
            )
        batch_axis.legend(loc="best")


# This function draws a confusion matrix heatmap for class-wise validation analysis.
def update_confusion_matrix(
    axis: plt.Axes,
    class_names: List[str],
    true_labels: List[int],
    predicted_labels: List[int],
) -> np.ndarray:
    axis.clear()
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )
    axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title("Validation Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=25, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size > 0 else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
                fontsize=10,
                fontweight="bold",
            )
    return matrix


# This function refreshes the prediction-monitoring figure with sample validation results.
def update_prediction_monitor(
    sample_figure: plt.Figure,
    sample_axes: np.ndarray,
    class_names: List[str],
    epoch_index: int,
    epoch_result: EpochResult,
) -> None:
    flat_axes = sample_axes.flatten()
    for axis in flat_axes:
        axis.clear()
        axis.axis("off")

    for index, axis in enumerate(flat_axes):
        if index >= len(epoch_result.sample_images):
            axis.set_title("No sample available")
            continue

        axis.imshow(epoch_result.sample_images[index], cmap="gray")
        true_label = class_names[epoch_result.sample_true_labels[index]]
        predicted_label = class_names[epoch_result.sample_predicted_labels[index]]
        confidence = epoch_result.sample_confidences[index] * 100
        axis.set_title(
            f"Epoch {epoch_index}\nTrue: {true_label}\nPred: {predicted_label} ({confidence:.1f}%)",
            fontsize=10,
        )
        axis.axis("off")

    sample_figure.tight_layout(rect=[0, 0.02, 1, 0.95])


# This function renders original images beside Grad-CAM overlays for attention analysis.
def update_gradcam_monitor(
    gradcam_figure: plt.Figure,
    gradcam_axes: np.ndarray,
    class_names: List[str],
    epoch_index: int,
    epoch_result: EpochResult,
) -> None:
    axes = np.atleast_2d(gradcam_axes)
    for row in axes:
        for axis in row:
            axis.clear()
            axis.axis("off")

    max_rows = axes.shape[0]
    for index in range(min(len(epoch_result.sample_images), max_rows)):
        original_axis = axes[index, 0]
        gradcam_axis = axes[index, 1]

        image = epoch_result.sample_images[index]
        overlay = epoch_result.gradcam_overlays[index] if index < len(epoch_result.gradcam_overlays) else image
        true_label = class_names[epoch_result.sample_true_labels[index]]
        predicted_label = class_names[epoch_result.sample_predicted_labels[index]]
        confidence = epoch_result.sample_confidences[index] * 100

        original_axis.imshow(image)
        original_axis.set_title(f"Epoch {epoch_index} Sample {index + 1}\nTrue: {true_label}", fontsize=10)
        original_axis.axis("off")

        gradcam_axis.imshow(overlay)
        gradcam_axis.set_title(
            f"Pred: {predicted_label} ({confidence:.1f}%)\nGrad-CAM focus overlay",
            fontsize=10,
        )
        gradcam_axis.axis("off")

    gradcam_figure.tight_layout(rect=[0, 0.02, 1, 0.97])


# This function saves the dashboard, confusion matrix, and prediction-monitor figures.
def save_plots(
    screenshots_dir: Path,
    epoch_index: int,
    context: VisualizationContext,
) -> None:
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    context.dashboard_figure.savefig(
        screenshots_dir / f"epoch_{epoch_index}_loss.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.dashboard_figure.savefig(
        screenshots_dir / f"epoch_{epoch_index}_dashboard.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.confusion_figure.savefig(
        screenshots_dir / f"epoch_{epoch_index}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.sample_figure.savefig(
        screenshots_dir / f"epoch_{epoch_index}_predictions.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.gradcam_figure.savefig(
        screenshots_dir / f"epoch_{epoch_index}_gradcam.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.dashboard_figure.savefig(
        screenshots_dir / "latest_training.png",
        dpi=300,
        bbox_inches="tight",
    )
    context.gradcam_figure.savefig(
        screenshots_dir / "latest_gradcam.png",
        dpi=300,
        bbox_inches="tight",
    )


# This function refreshes the live figures without blocking the training loop.
def refresh_live_plots() -> None:
    plt.draw()
    plt.pause(0.001)


# This function builds a normalized Grad-CAM heatmap for a single sample tensor.
def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    target_layer = model.features[-1]
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(_: nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output.detach())

    def backward_hook(_: nn.Module, __: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0].detach())

    was_training = model.training
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        model_device = next(model.parameters()).device
        input_batch = input_tensor.unsqueeze(0).to(model_device)
        output = model(input_batch)
        score = output[0, target_class]
        score.backward()

        activation_map = activations[0]
        gradient_map = gradients[0]
        weights = gradient_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation_map, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


# This function overlays a Grad-CAM heatmap on top of a denormalized image.
def create_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    color_heatmap = plt.cm.jet(heatmap)[..., :3]
    overlay = 0.55 * image + 0.45 * color_heatmap
    return np.clip(overlay, 0, 1)


# This function enriches validation samples with Grad-CAM attention maps for documentation.
def attach_gradcam_visuals(model: nn.Module, epoch_result: EpochResult) -> None:
    if not epoch_result.sample_tensors:
        return

    gradcam_heatmaps: List[np.ndarray] = []
    gradcam_overlays: List[np.ndarray] = []
    for image_tensor, predicted_label, display_image in zip(
        epoch_result.sample_tensors,
        epoch_result.sample_predicted_labels,
        epoch_result.sample_images,
    ):
        cam = generate_gradcam(model, image_tensor, predicted_label)
        gradcam_heatmaps.append(cam)
        gradcam_overlays.append(create_gradcam_overlay(display_image, cam))

    epoch_result.gradcam_heatmaps = gradcam_heatmaps
    epoch_result.gradcam_overlays = gradcam_overlays


# This function summarizes confusion-matrix performance in readable text.
def summarize_class_performance(matrix: np.ndarray, class_names: List[str]) -> List[str]:
    summaries: List[str] = []
    for class_index, class_name in enumerate(class_names):
        class_total = int(matrix[class_index].sum())
        correct = int(matrix[class_index, class_index])
        recall = (correct / class_total * 100) if class_total > 0 else 0.0
        mistakes = matrix[class_index].copy()
        mistakes[class_index] = 0
        if mistakes.sum() > 0:
            confused_with = class_names[int(np.argmax(mistakes))]
            confusion_note = f"Most confusion was with '{confused_with}'."
        else:
            confusion_note = "No misclassifications were recorded for this class in this epoch."
        summaries.append(
            f"- `{class_name}`: {correct}/{class_total} correct in validation ({recall:.1f}% recall). {confusion_note}"
        )
    return summaries


# This function creates a short interpretation for each monitored image.
def interpret_sample(
    class_names: List[str],
    sample_index: int,
    epoch_result: EpochResult,
) -> str:
    true_label = class_names[epoch_result.sample_true_labels[sample_index]]
    predicted_label = class_names[epoch_result.sample_predicted_labels[sample_index]]
    confidence = epoch_result.sample_confidences[sample_index] * 100
    heatmap = epoch_result.gradcam_heatmaps[sample_index] if sample_index < len(epoch_result.gradcam_heatmaps) else None

    if heatmap is not None:
        focus_strength = float(heatmap.max())
        highlighted_area = float((heatmap > 0.6).mean() * 100)
        attention_note = (
            f"The Grad-CAM map shows a peak activation of {focus_strength:.2f} with about "
            f"{highlighted_area:.1f}% of the image strongly highlighted, which helps show how concentrated "
            "or diffuse the model's attention was."
        )
    else:
        attention_note = "No Grad-CAM map was generated for this sample."

    if true_label == predicted_label:
        outcome_note = "The prediction matched the validation label, so this image is useful for showing correct class-specific attention."
    else:
        outcome_note = (
            "The prediction did not match the validation label, so this image is useful for discussing where the model's "
            "attention may have supported a class confusion."
        )

    return (
        f"- Sample {sample_index + 1}: true label `{true_label}`, predicted `{predicted_label}` "
        f"with {confidence:.1f}% confidence. {outcome_note} {attention_note}"
    )


# This function writes an epoch-by-epoch markdown report for research discussion.
def write_epoch_report(
    reports_dir: Path,
    epoch_index: int,
    history: TrainingHistory,
    class_names: List[str],
    confusion: np.ndarray,
    epoch_result: EpochResult,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.train_losses[-1]
    val_loss = history.val_losses[-1]
    train_acc = history.train_accuracies[-1] * 100
    val_acc = history.val_accuracies[-1] * 100

    if len(history.val_losses) > 1:
        val_loss_change = history.val_losses[-2] - history.val_losses[-1]
        val_acc_change = (history.val_accuracies[-1] - history.val_accuracies[-2]) * 100
        progress_note = (
            f"Compared with the previous epoch, validation loss changed by {val_loss_change:+.4f} "
            f"and validation accuracy changed by {val_acc_change:+.2f} percentage points."
        )
    else:
        progress_note = "This is the first epoch, so it establishes the baseline behavior of the model."

    if train_acc - val_acc > 8:
        generalization_note = "The train-validation gap is becoming noticeable, which may indicate emerging overfitting."
    elif val_acc >= train_acc:
        generalization_note = "Validation performance is tracking closely with training performance, which is encouraging for generalization."
    else:
        generalization_note = "Training performance is modestly ahead of validation performance, which is typical during fitting."

    class_summaries = summarize_class_performance(confusion, class_names)
    sample_notes = [interpret_sample(class_names, index, epoch_result) for index in range(len(epoch_result.sample_images))]

    report_lines = [
        f"# Epoch {epoch_index} Research Notes",
        "",
        "## Metric Summary",
        f"- Training loss: {train_loss:.4f}",
        f"- Validation loss: {val_loss:.4f}",
        f"- Training accuracy: {train_acc:.2f}%",
        f"- Validation accuracy: {val_acc:.2f}%",
        f"- {progress_note}",
        f"- {generalization_note}",
        "",
        "## Figure References",
        f"- Loss/dashboard figure: `screenshots/epoch_{epoch_index}_dashboard.png`",
        f"- Confusion matrix: `screenshots/epoch_{epoch_index}_confusion_matrix.png`",
        f"- Prediction monitoring: `screenshots/epoch_{epoch_index}_predictions.png`",
        f"- Grad-CAM attention figure: `screenshots/epoch_{epoch_index}_gradcam.png`",
        "",
        "## Confusion Matrix Discussion",
        "These counts show how the validation set is distributed across correct predictions and class confusions:",
        *class_summaries,
        "",
        "## Interpreting The Monitored Images",
        "The monitored images are useful for discussing whether the network's decision appears focused and consistent across examples.",
    ]

    if sample_notes:
        report_lines.extend(sample_notes)
    else:
        report_lines.append("- No monitored images were captured in this epoch.")

    report_lines.extend(
        [
            "",
            "## Significance For Research Discussion",
            "- The loss and accuracy curves help show whether the network is still learning, stabilizing, or beginning to overfit.",
            "- The confusion matrix highlights which disease stages are easiest or hardest for the model to separate.",
            "- The prediction monitor provides concrete examples of correct and incorrect decisions that can be discussed qualitatively.",
            "- The Grad-CAM overlays help relate model attention to image regions, which can support interpretability-oriented discussion.",
        ]
    )

    report_text = "\n".join(report_lines) + "\n"
    epoch_report_path = reports_dir / f"epoch_{epoch_index}_report.md"
    latest_report_path = reports_dir / "latest_report.md"
    epoch_report_path.write_text(report_text, encoding="utf-8")
    latest_report_path.write_text(report_text, encoding="utf-8")


# This function runs one full pass over a dataloader and returns metrics plus outputs.
def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    history: TrainingHistory | None = None,
    monitor_samples: bool = False,
    max_monitor_images: int = NUM_MONITOR_IMAGES,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    confidences: List[float] = []
    sample_images: List[np.ndarray] = []
    sample_true_labels: List[int] = []
    sample_predicted_labels: List[int] = []
    sample_confidences: List[float] = []
    sample_tensors: List[torch.Tensor] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=torch.cuda.is_available())
        labels = labels.to(device, non_blocking=torch.cuda.is_available())

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
                if history is not None:
                    history.batch_losses.append(loss.item())

        probabilities = torch.softmax(outputs, dim=1)
        confidence_values, predictions = torch.max(probabilities, dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        true_labels.extend(labels.detach().cpu().tolist())
        predicted_labels.extend(predictions.detach().cpu().tolist())
        confidences.extend(confidence_values.detach().cpu().tolist())

        if monitor_samples and len(sample_images) < max_monitor_images:
            remaining_slots = max_monitor_images - len(sample_images)
            for image_tensor, true_label, pred_label, confidence in zip(
                inputs[:remaining_slots],
                labels[:remaining_slots],
                predictions[:remaining_slots],
                confidence_values[:remaining_slots],
            ):
                sample_images.append(denormalize_image(image_tensor))
                sample_tensors.append(image_tensor.detach().cpu())
                sample_true_labels.append(int(true_label.detach().cpu().item()))
                sample_predicted_labels.append(int(pred_label.detach().cpu().item()))
                sample_confidences.append(float(confidence.detach().cpu().item()))

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return EpochResult(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
        sample_images=sample_images,
        sample_true_labels=sample_true_labels,
        sample_predicted_labels=sample_predicted_labels,
        sample_confidences=sample_confidences,
        sample_tensors=sample_tensors,
    )


# This function saves the best model checkpoint for later Flask inference.
def save_checkpoint(model: nn.Module, class_names: List[str], output_path: Path) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    }
    torch.save(checkpoint, output_path)


# This function handles the full training loop, live dashboard, and checkpoint selection.
def train_model() -> None:
    set_seed(RANDOM_SEED)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    train_loader, val_loader, class_names = create_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=RANDOM_SEED,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = TrainingHistory()
    visualization = initialize_visualization()

    best_val_accuracy = -1.0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        epoch_index = epoch + 1
        train_result = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            history=history,
            monitor_samples=False,
        )
        val_result = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            history=None,
            monitor_samples=(epoch_index % PREDICTION_MONITOR_INTERVAL == 0) or (epoch_index == 1),
        )

        history.train_losses.append(train_result.loss)
        history.val_losses.append(val_result.loss)
        history.train_accuracies.append(train_result.accuracy)
        history.val_accuracies.append(val_result.accuracy)

        scheduler.step(val_result.loss)

        update_training_plot(visualization.dashboard_axes, history, epoch_index)
        confusion = update_confusion_matrix(
            visualization.dashboard_axes[1, 0],
            class_names,
            val_result.true_labels,
            val_result.predicted_labels,
        )
        update_confusion_matrix(
            visualization.confusion_axis,
            class_names,
            val_result.true_labels,
            val_result.predicted_labels,
        )

        if val_result.sample_images:
            attach_gradcam_visuals(model, val_result)
        update_prediction_monitor(
            visualization.sample_figure,
            visualization.sample_axes,
            class_names,
            epoch_index,
            val_result,
        )
        update_gradcam_monitor(
            visualization.gradcam_figure,
            visualization.gradcam_axes,
            class_names,
            epoch_index,
            val_result,
        )

        visualization.dashboard_figure.tight_layout(rect=[0, 0.02, 1, 0.96])
        visualization.confusion_figure.tight_layout(rect=[0, 0.02, 1, 0.95])
        visualization.gradcam_figure.tight_layout(rect=[0, 0.02, 1, 0.97])
        refresh_live_plots()
        save_plots(SCREENSHOTS_DIR, epoch_index, visualization)
        write_epoch_report(REPORTS_DIR, epoch_index, history, class_names, confusion, val_result)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch_index}/{NUM_EPOCHS}] "
            f"Train Loss: {train_result.loss:.4f} | Train Acc: {train_result.accuracy:.4f} | "
            f"Val Loss: {val_result.loss:.4f} | Val Acc: {val_result.accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_result.accuracy > best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(model, class_names, MODEL_PATH)
            print(f"Best model updated and saved to: {MODEL_PATH.resolve()}")

    model.load_state_dict(best_model_weights)
    save_checkpoint(model, class_names, MODEL_PATH)

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final best model saved to: {MODEL_PATH.resolve()}")
    print(f"Figures saved to: {SCREENSHOTS_DIR.resolve()}")
    print(f"Reports saved to: {REPORTS_DIR.resolve()}")
    print(f"Class mapping used by ImageFolder: {class_names}")

    plt.ioff()
    plt.show(block=False)


if __name__ == "__main__":
    train_model()
