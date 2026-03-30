# MIRA

**Medical Image Recognition for Alzheimer's**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-0b7285.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet%20Pipeline-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-12b5cb.svg)](#)

MIRA is a Flask + PyTorch web application for classifying Alzheimer's disease stages from MRI images using **EfficientNet-B0**. It combines a training pipeline, a responsive web dashboard, phone-photo preprocessing, and research-friendly outputs such as confusion matrices, Grad-CAM visualizations, and epoch-by-epoch notes.

Research/education only.

## Highlights

- EfficientNet-B0 transfer learning with PyTorch
- Flask web app for MRI upload and stage prediction
- preprocessing support for direct image files and phone photos of scans
- responsive medical-dashboard style UI
- confidence-aware warnings for uncertain predictions
- training dashboard with:
  - loss curves
  - accuracy curves
  - confusion matrix
  - batch-loss tracking
  - prediction monitoring
  - Grad-CAM attention maps
- markdown research notes saved per epoch

## Quick Start

### 1. Create a local virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Prepare the dataset

Store the MRI images in:

```text
data/
|-- Mild Impairment/
|-- Moderate Impairment/
|-- No Impairment/
`-- Very Mild Impairment/
```

Expected label set:

```python
[
  "No Impairment",
  "Very Mild Impairment",
  "Mild Impairment",
  "Moderate Impairment",
]
```

### 3. Train the model

```powershell
python train.py
```

### 4. Run the web app

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## What MIRA Does

### Training Pipeline

- loads the dataset using `torchvision.datasets.ImageFolder`
- applies augmentation and ImageNet normalization
- trains EfficientNet-B0 with transfer learning
- saves the best checkpoint to `alz_model.pth`
- updates live matplotlib figures during training
- exports research figures and markdown notes to `screenshots/`

### Web Inference Workflow

- accepts JPG, JPEG, PNG, and BMP images
- preprocesses uploads before inference
- saves the processed image to `static/processed_uploads/`
- displays prediction confidence and warning states
- works on CPU or GPU depending on local availability

### Phone-Photo Support

If the upload is a phone photo of a scan, MIRA attempts to improve it by:

- correcting orientation
- cropping toward the scan region
- enhancing grayscale contrast
- padding to a square frame
- resizing to the model input size

## Outputs

### Training Outputs

- `alz_model.pth`
- `screenshots/latest_training.png`
- `screenshots/latest_gradcam.png`
- `screenshots/reports/latest_report.md`

### Web App Outputs

- processed uploads saved in `static/processed_uploads/`
- confidence-based result messaging in the UI

## Project Structure

```text
MIRA/
|-- app.py
|-- train.py
|-- requirements.txt
|-- README.md
|-- templates/
|   |-- index.html
|   `-- result.html
|-- static/
|   |-- style.css
|   `-- processed_uploads/
|-- screenshots/
|   `-- reports/
`-- data/
    |-- Mild Impairment/
    |-- Moderate Impairment/
    |-- No Impairment/
    `-- Very Mild Impairment/
```

## Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Pillow
- NumPy

## Git Notes

The repo ignores local-only artifacts such as:

- `.venv/`
- `data/`
- `screenshots/`
- `static/processed_uploads/`
- `alz_model.pth`

This keeps GitHub focused on the source code and project files rather than large local assets.

## Roadmap

- add persistent prediction history
- add drag-and-drop upload interactions
- add exportable experiment summaries
- refine mobile interactions further
