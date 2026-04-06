# MIRA

Medical Image Recognition for Alzheimer's

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-0b7285.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet%20Pipeline-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-12b5cb.svg)](#)

MIRA is a Flask + PyTorch web application for classifying Alzheimer's disease stages from MRI images using an EfficientNet-B0 model. It combines model training, MRI upload inference, phone-photo preprocessing, and research outputs such as training dashboards, confusion matrices, Grad-CAM visualizations, and markdown reports.

This project is intended for research and education only.

## Overview

MIRA supports two main workflows:

1. Training a multi-class Alzheimer's stage classifier from MRI image folders in `data/`
2. Running a web app that lets you upload an MRI image and review the prediction result in a cleaner multi-page interface

The app now uses separate pages instead of a single scrolling dashboard:

- `Home` for the overview and system summary
- `Upload` for MRI submission
- `Results` for saved training visualizations
- `About` for project context
- `Prediction Result` for the outcome of a specific uploaded image

## Features

### Model and Training

- EfficientNet-B0 classification pipeline using PyTorch and Torchvision
- Transfer-learning-based training flow
- Image augmentation and ImageNet normalization
- Best-model checkpoint saving to `alz_model.pth`
- Training dashboard generation with loss curves
- Training dashboard generation with accuracy curves
- Training dashboard generation with batch loss monitoring
- Training dashboard generation with confusion matrix export
- Training dashboard generation with prediction monitoring
- Training dashboard generation with Grad-CAM attention visualizations
- Markdown report export for experiment notes

### Web App

- Flask-based web UI
- Dedicated pages for Home, Upload, Results, and About
- Sticky top navigation on desktop and mobile-friendly bottom navigation
- MRI upload form with loading state
- Prediction result screen with confidence feedback
- Saved processed-image preview after inference

### Image Preprocessing

- Accepts `.jpg`, `.jpeg`, `.png`, and `.bmp`
- Handles direct image uploads and phone photos of scans
- Automatically corrects image orientation
- Automatically estimates and crops the scan region
- Automatically enhances grayscale contrast
- Automatically sharpens and normalizes the image
- Automatically pads to a square frame
- Automatically resizes to the model input size

## Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

## Requirements

The project currently depends on:

- Flask
- matplotlib
- numpy
- Pillow
- scikit-learn
- torch
- torchvision

Install them with:

```powershell
python -m pip install -r requirements.txt
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Prepare the dataset

Place MRI images in the `data/` directory using class folders like this:

```text
data/
|-- Mild Impairment/
|-- Moderate Impairment/
|-- No Impairment/
`-- Very Mild Impairment/
```

Expected class labels:

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

This creates:

- `alz_model.pth`
- updated visual outputs in `screenshots/`
- report files in `screenshots/reports/`

### 4. Run the web app

```powershell
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Web Routes

The Flask app currently exposes these main routes:

- `/` -> Home page
- `/upload` -> MRI upload page
- `/analytics` -> saved training visualizations page
- `/about` -> project summary page
- `/predict` -> POST route for inference
- `/artifacts/<path:filename>` -> training image artifacts
- `/assets/<path:filename>` -> custom assets such as the logo

## Inference Flow

When a user uploads an image:

1. The file type is validated
2. The image is temporarily saved
3. The image is preprocessed for inference
4. The processed image is saved to `static/processed_uploads/`
5. The model predicts the Alzheimer's stage
6. Confidence messaging is generated
7. The result page shows the predicted class
8. The result page shows the confidence score
9. The result page shows the confidence label
10. The result page shows the processed preview image

## Training Outputs

Typical outputs created during or after training:

- `alz_model.pth`
- `screenshots/latest_training.png`
- `screenshots/latest_gradcam.png`
- `screenshots/epoch_*_confusion_matrix.png`
- `screenshots/reports/latest_report.md`

## Project Structure

```text
MIRA/
|-- app.py
|-- train.py
|-- requirements.txt
|-- README.md
|-- assets/
|   `-- img/
|       `-- mira.png
|-- templates/
|   |-- index.html
|   |-- upload.html
|   |-- analytics.html
|   |-- about.html
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

## Notes About Local Files

Large or local-only files should usually stay out of Git, including:

- virtual environments such as `.venv/`
- dataset contents in `data/`
- generated screenshots and reports if you do not want large artifacts committed
- processed uploads in `static/processed_uploads/`
- trained model weights such as `alz_model.pth`

Review `.gitignore` before pushing if you want to keep GitHub focused on source code and documentation.

## Limitations

- This is not a clinical diagnostic tool
- Prediction quality depends on data quality and model training quality
- The app assumes a trained model checkpoint already exists
- Results should be treated as research outputs, not medical decisions

## Roadmap Ideas

- persist upload and prediction history
- add downloadable prediction summaries
- improve analytics browsing for multiple runs
- add model/version metadata to the UI
- refine responsive behavior further
