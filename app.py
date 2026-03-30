"""
app.py
Purpose: Run a Flask web application that accepts MRI image uploads and
predicts the Alzheimer's disease stage using a trained EfficientNet-B0 model.

Research/education only.

Recommended setup before running:
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install -r requirements.txt

Run:
    python app.py
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b0


MODEL_PATH = Path("alz_model.pth")
PROCESSED_UPLOADS_DIR = Path("static") / "processed_uploads"
SCREENSHOTS_DIR = Path("screenshots")
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# This list is used as a fallback if the checkpoint does not include class names.
DEFAULT_CLASSES = [
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment",
]


app = Flask(__name__)
app.config["SECRET_KEY"] = "alzheimer-demo-secret-key"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names: List[str] = DEFAULT_CLASSES


# This function returns the most recently modified file that matches a pattern.
def get_latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matching_files = [path for path in directory.glob(pattern) if path.is_file()]
    if not matching_files:
        return None
    return max(matching_files, key=lambda path: path.stat().st_mtime)


# This function builds the analytics panel data from the saved training artifacts.
def get_analytics_assets() -> List[dict]:
    latest_training = SCREENSHOTS_DIR / "latest_training.png"
    latest_gradcam = SCREENSHOTS_DIR / "latest_gradcam.png"
    latest_confusion = get_latest_matching_file(SCREENSHOTS_DIR, "epoch_*_confusion_matrix.png")

    assets = [
        {
            "title": "Training Curves",
            "pill": "Live",
            "description": "Latest saved training dashboard with loss and accuracy curves.",
            "path": latest_training if latest_training.exists() else None,
        },
        {
            "title": "Confusion Matrix",
            "pill": "Epoch",
            "description": "Most recent validation confusion matrix exported during training.",
            "path": latest_confusion,
        },
        {
            "title": "Attention Map",
            "pill": "Grad-CAM",
            "description": "Latest Grad-CAM overlay showing where the model focused.",
            "path": latest_gradcam if latest_gradcam.exists() else None,
        },
    ]

    for asset in assets:
        path = asset["path"]
        if path is None:
            asset["image_url"] = None
            asset["open_url"] = None
        else:
            relative_path = path.relative_to(SCREENSHOTS_DIR).as_posix()
            asset["image_url"] = url_for("training_artifact", filename=relative_path)
            asset["open_url"] = asset["image_url"]

    return assets


# This function returns the preprocessing pipeline for model inference.
def get_inference_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# This function verifies whether an uploaded file uses a supported image extension.
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# This function estimates the main scan area so phone photos can be cropped tighter.
def detect_scan_bbox(image: Image.Image) -> Tuple[int, int, int, int]:
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.float32)
    height, width = pixels.shape

    border_pixels = np.concatenate([pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]])
    background_level = float(np.median(border_pixels))
    contrast_scale = max(12.0, float(np.std(border_pixels) * 2.5))

    foreground_mask = np.abs(pixels - background_level) > contrast_scale
    if foreground_mask.mean() < 0.01:
        dynamic_range = float(np.percentile(pixels, 90) - np.percentile(pixels, 10))
        contrast_scale = max(10.0, dynamic_range * 0.18)
        if background_level > float(pixels.mean()):
            foreground_mask = pixels < (background_level - contrast_scale)
        else:
            foreground_mask = pixels > (background_level + contrast_scale)

    if foreground_mask.mean() < 0.005:
        return (0, 0, width, height)

    coordinates = np.argwhere(foreground_mask)
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    pad_y = max(10, int((bottom - top + 1) * 0.08))
    pad_x = max(10, int((right - left + 1) * 0.08))

    left = max(0, int(left - pad_x))
    top = max(0, int(top - pad_y))
    right = min(width, int(right + pad_x))
    bottom = min(height, int(bottom + pad_y))
    return (left, top, right, bottom)


# This function pads a cropped scan to a square frame for more stable inference.
def pad_to_square(image: Image.Image, fill_color: int = 0) -> Image.Image:
    width, height = image.size
    if width == height:
        return image

    side_length = max(width, height)
    square_image = Image.new("L", (side_length, side_length), color=fill_color)
    square_image.paste(image, ((side_length - width) // 2, (side_length - height) // 2))
    return square_image


# This function cleans a phone photo and saves the processed image used for inference.
def process_uploaded_image(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    cropped_image = image.crop(detect_scan_bbox(image))

    grayscale = ImageOps.grayscale(cropped_image)
    enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
    enhanced = ImageOps.equalize(enhanced)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    square_image = pad_to_square(enhanced, fill_color=0)
    final_image = square_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR).convert("RGB")

    processed_path = output_dir / f"{uuid.uuid4().hex}.png"
    final_image.save(processed_path, format="PNG")
    return processed_path


# This function loads and preprocesses an image for model inference.
def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


# This function converts a raw confidence score into UI-friendly status messaging.
def build_confidence_feedback(confidence_score: float) -> Tuple[str, str, str]:
    if confidence_score >= 85.0:
        return (
            "High confidence",
            "The model is relatively confident on this processed input.",
            "confidence-high",
        )
    if confidence_score >= 65.0:
        return (
            "Moderate confidence",
            "The prediction is usable, but it would be wise to review the processed image and result carefully.",
            "confidence-medium",
        )
    return (
        "Low confidence",
        "The model is uncertain on this input. Review the processed image closely and consider trying a cleaner or more direct scan image.",
        "confidence-low",
    )


# This function creates a model architecture that matches training.
def build_model(num_classes: int) -> torch.nn.Module:
    model_instance = efficientnet_b0(weights=None)
    in_features = model_instance.classifier[1].in_features
    model_instance.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model_instance


# This function loads the saved checkpoint and prepares the model for inference.
def load_trained_model(model_path: Path) -> Tuple[torch.nn.Module, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path.resolve()}. Run `python train.py` first."
        )

    checkpoint = torch.load(model_path, map_location="cpu")

    loaded_class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
    num_classes = len(loaded_class_names)
    model_instance = build_model(num_classes=num_classes)

    if "model_state_dict" in checkpoint:
        model_instance.load_state_dict(checkpoint["model_state_dict"])
    else:
        # This fallback supports checkpoints that only store a raw state dict.
        model_instance.load_state_dict(checkpoint)

    model_instance.to(device)
    model_instance.eval()
    return model_instance, loaded_class_names


# This function runs prediction and returns both the label and confidence score.
def predict_image(image_path: Path) -> Tuple[str, float]:
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_label = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100.0
    return predicted_label, confidence_score


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", analytics_assets=get_analytics_assets())


@app.route("/artifacts/<path:filename>", methods=["GET"])
def training_artifact(filename: str):
    return send_from_directory(SCREENSHOTS_DIR.resolve(), filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Please choose an image file before submitting.")
        return redirect(url_for("index"))

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        flash("No file selected. Please upload an image.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Please upload a JPG, JPEG, PNG, or BMP image.")
        return redirect(url_for("index"))

    temp_file_path: Path | None = None
    processed_file_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_file_path = Path(temp_file.name)

        processed_file_path = process_uploaded_image(temp_file_path, PROCESSED_UPLOADS_DIR)
        prediction, confidence = predict_image(processed_file_path)
        confidence_label, confidence_message, confidence_class = build_confidence_feedback(confidence)
        return render_template(
            "result.html",
            prediction=prediction,
            confidence=f"{confidence:.2f}",
            confidence_label=confidence_label,
            confidence_message=confidence_message,
            confidence_class=confidence_class,
            processed_image_url=url_for("static", filename=f"processed_uploads/{processed_file_path.name}"),
            processed_filename=processed_file_path.name,
        )
    except UnidentifiedImageError:
        flash("The uploaded file is not a valid image.")
        return redirect(url_for("index"))
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))
    finally:
        if temp_file_path is not None and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


if __name__ == "__main__":
    PROCESSED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    model, class_names = load_trained_model(MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
