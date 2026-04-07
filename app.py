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

import logging
import os
import secrets
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch
from flask import Flask, abort, flash, redirect, render_template, request, send_file, send_from_directory, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b0


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "alz_model.pth"
PROCESSED_UPLOADS_DIR = BASE_DIR / "static" / "processed_uploads"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
ASSETS_DIR = BASE_DIR / "assets"
CERTS_DIR = BASE_DIR / "certs"
STATIC_DIR = BASE_DIR / "static"
DOCS_STATIC_DIR = STATIC_DIR / "docs"
IMAGE_SIZE = 224
DEFAULT_MAX_UPLOAD_MB = 10
APP_HOST = os.getenv("MIRA_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("MIRA_PORT", "5000"))
USE_HTTPS = os.getenv("MIRA_USE_HTTPS", "1").lower() in {"1", "true", "yes", "on"}
APP_DEBUG = os.getenv("MIRA_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_CLASSES = [
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment",
]


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if APP_DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_local_ca_path() -> Path | None:
    candidates = [
        Path(os.getenv("MIRA_LOCAL_CA_PATH", "")),
        Path.home() / "AppData" / "Local" / "mkcert" / "rootCA.pem",
        Path.home() / "Library" / "Application Support" / "mkcert" / "rootCA.pem",
        BASE_DIR / "rootCA.pem",
        CERTS_DIR / "rootCA.pem",
    ]

    for candidate in candidates:
        if not str(candidate):
            continue
        if candidate.is_file():
            return candidate
    return None


def get_latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matching_files = [path for path in directory.glob(pattern) if path.is_file()]
    if not matching_files:
        return None
    return max(matching_files, key=lambda path: path.stat().st_mtime)


@lru_cache(maxsize=1)
def get_model_bundle() -> Tuple[torch.nn.Module, List[str]]:
    return load_trained_model(MODEL_PATH)


def build_install_context() -> dict:
    local_ca_path = get_local_ca_path()
    return {
        "local_ca_available": local_ca_path is not None,
        "local_ca_download_url": url_for("download_local_ca") if local_ca_path else None,
        "host_url": request.host_url.rstrip("/"),
        "install_page_url": url_for("install_page"),
        "documentation_markdown_url": url_for("static", filename="docs/MIRA-Production-Guide.md"),
        "documentation_pdf_url": url_for("static", filename="docs/mira-production-guide.pdf"),
        "model_ready": MODEL_PATH.exists(),
    }


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


def get_inference_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def process_uploaded_image(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    # Keep inference preprocessing aligned with the training pipeline.
    final_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    processed_path = output_dir / f"{uuid.uuid4().hex}.png"
    final_image.save(processed_path, format="PNG")
    return processed_path


def save_preview_image(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    preview_image = image.copy()
    preview_image.thumbnail((640, 640), Image.Resampling.BILINEAR)

    preview_path = output_dir / f"{uuid.uuid4().hex}_preview.png"
    preview_image.save(preview_path, format="PNG")
    return preview_path


def preprocess_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def build_confidence_feedback(confidence_score: float) -> Tuple[str, str, str]:
    if confidence_score >= 85.0:
        return ("High confidence", "The model is relatively confident on this processed input.", "confidence-high")
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


def build_model(num_classes: int) -> torch.nn.Module:
    model_instance = efficientnet_b0(weights=None)
    in_features = model_instance.classifier[1].in_features
    model_instance.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model_instance


def load_trained_model(model_path: Path) -> Tuple[torch.nn.Module, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()}. Run `python train.py` first.")

    checkpoint = torch.load(model_path, map_location="cpu")
    loaded_class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
    model_instance = build_model(num_classes=len(loaded_class_names))

    if "model_state_dict" in checkpoint:
        model_instance.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_instance.load_state_dict(checkpoint)

    model_instance.to(device)
    model_instance.eval()
    logger.info("Model loaded from %s on %s", model_path, device)
    return model_instance, loaded_class_names


def predict_image(image_path: Path) -> Tuple[str, float]:
    model, class_names = get_model_bundle()
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_label = class_names[predicted_index.item()]
    return predicted_label, confidence.item() * 100.0


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("MIRA_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=int(os.getenv("MIRA_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB))) * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=USE_HTTPS,
        TEMPLATES_AUTO_RELOAD=APP_DEBUG,
    )

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        flash(f"Upload too large. Keep files under {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB.")
        return redirect(url_for("upload_page"))

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.exception("Unhandled server error: %s", error)
        return render_template("about.html", server_error=True, **build_install_context()), 500

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", **build_install_context())

    @app.route("/upload", methods=["GET"])
    def upload_page():
        return render_template("upload.html", **build_install_context())

    @app.route("/analytics", methods=["GET"])
    def analytics_page():
        return render_template("analytics.html", analytics_assets=get_analytics_assets(), **build_install_context())

    @app.route("/about", methods=["GET"])
    def about_page():
        return render_template("about.html", server_error=False, **build_install_context())

    @app.route("/install", methods=["GET"])
    def install_page():
        return render_template("install.html", **build_install_context())

    @app.route("/downloads/local-ca", methods=["GET"])
    def download_local_ca():
        local_ca_path = get_local_ca_path()
        if local_ca_path is None:
            abort(404)
        return send_file(local_ca_path, as_attachment=True, download_name="mira-local-rootCA.pem")

    @app.route("/artifacts/<path:filename>", methods=["GET"])
    def training_artifact(filename: str):
        return send_from_directory(SCREENSHOTS_DIR.resolve(), filename)

    @app.route("/manifest.webmanifest", methods=["GET"])
    def web_manifest():
        return send_from_directory(STATIC_DIR.resolve(), "manifest.webmanifest", mimetype="application/manifest+json")

    @app.route("/service-worker.js", methods=["GET"])
    def service_worker():
        return send_from_directory(STATIC_DIR.resolve(), "service-worker.js", mimetype="application/javascript")

    @app.route("/assets/<path:filename>", methods=["GET"])
    def asset_file(filename: str):
        return send_from_directory(ASSETS_DIR.resolve(), filename)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            flash("Please choose an image file before submitting.")
            return redirect(url_for("upload_page"))

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            flash("No file selected. Please upload an image.")
            return redirect(url_for("upload_page"))

        if not allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload a JPG, JPEG, PNG, or BMP image.")
            return redirect(url_for("upload_page"))

        temp_file_path: Path | None = None

        try:
            suffix = Path(uploaded_file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                uploaded_file.save(temp_file.name)
                temp_file_path = Path(temp_file.name)

            processed_file_path = process_uploaded_image(temp_file_path, PROCESSED_UPLOADS_DIR)
            preview_file_path = save_preview_image(temp_file_path, PROCESSED_UPLOADS_DIR)
            prediction, confidence = predict_image(processed_file_path)
            confidence_label, confidence_message, confidence_class = build_confidence_feedback(confidence)
            logger.info(
                "Prediction completed for upload %s with label '%s' at %.2f%% confidence",
                processed_file_path.name,
                prediction,
                confidence,
            )
            return render_template(
                "result.html",
                prediction=prediction,
                confidence=f"{confidence:.2f}",
                confidence_label=confidence_label,
                confidence_message=confidence_message,
                confidence_class=confidence_class,
                processed_image_url=url_for("static", filename=f"processed_uploads/{preview_file_path.name}"),
                processed_filename=processed_file_path.name,
                **build_install_context(),
            )
        except UnidentifiedImageError:
            flash("The uploaded file is not a valid image.")
            return redirect(url_for("upload_page"))
        except FileNotFoundError as exc:
            logger.exception("Model or dependency file missing during prediction")
            flash(str(exc))
            return redirect(url_for("upload_page"))
        except Exception:
            logger.exception("Prediction failed unexpectedly")
            flash("Prediction failed due to an unexpected server error. Check the application logs for details.")
            return redirect(url_for("upload_page"))
        finally:
            if temp_file_path is not None and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError:
                    logger.warning("Could not remove temporary upload file %s", temp_file_path)

    return app


def get_ssl_context():
    cert_file = CERTS_DIR / "mira-local.pem"
    key_file = CERTS_DIR / "mira-local-key.pem"

    if USE_HTTPS and cert_file.exists() and key_file.exists():
        return (str(cert_file), str(key_file))
    if USE_HTTPS:
        return "adhoc"
    return None


configure_logging()
app = create_app()


if __name__ == "__main__":
    PROCESSED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    ssl_context = get_ssl_context()
    protocol = "https" if ssl_context else "http"
    logger.info("Server ready at %s://127.0.0.1:%s and waiting for requests.", protocol, APP_PORT)
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        use_reloader=False,
        ssl_context=ssl_context,
    )

