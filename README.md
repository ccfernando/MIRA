# MIRA

Medical Image Recognition for Alzheimer's (MIRA) is a Flask + PyTorch web application for classifying Alzheimer's disease stages from MRI images using EfficientNet-B0.

This project includes:
- a training pipeline with transfer learning
- a responsive web interface for image upload and prediction
- preprocessing support for phone photos of MRI scans
- live training visualizations, confusion matrices, Grad-CAM, and research notes

Research/education only.

## Features

- EfficientNet-B0 training with PyTorch and `torchvision`
- dataset loading with `ImageFolder`
- train/validation split with per-epoch metrics
- best-model checkpoint saving to `alz_model.pth`
- Flask app for MRI upload and inference
- preprocessing pipeline for uploaded images before prediction
- processed upload preservation in `static/processed_uploads/`
- training dashboard with:
  - loss curves
  - accuracy curves
  - confusion matrix
  - batch-loss tracking
  - prediction monitoring
  - Grad-CAM visualizations
- epoch-by-epoch markdown reports for research discussion

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

## Dataset Layout

The training dataset should be stored locally in:

```text
data/
|-- Mild Impairment/
|-- Moderate Impairment/
|-- No Impairment/
`-- Very Mild Impairment/
```

Expected labels:

```python
[
  "No Impairment",
  "Very Mild Impairment",
  "Mild Impairment",
  "Moderate Impairment",
]
```

## Setup

Create and activate a project-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train The Model

Run:

```powershell
python train.py
```

Training outputs:
- best model checkpoint: `alz_model.pth`
- saved figures: `screenshots/`
- latest dashboard image: `screenshots/latest_training.png`
- latest Grad-CAM image: `screenshots/latest_gradcam.png`
- epoch reports: `screenshots/reports/`

The training script includes:
- transfer learning with EfficientNet-B0
- real-time matplotlib dashboard updates
- validation confusion matrix
- prediction preview panel
- Grad-CAM attention overlays
- markdown logs for discussing results and image significance

## Run The Web App

After training, start the Flask app:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Web App Behavior

The web app:
- accepts MRI images in JPG, JPEG, PNG, or BMP format
- preprocesses the upload before inference
- saves the processed image to `static/processed_uploads/`
- loads the trained checkpoint from `alz_model.pth`
- runs inference on CPU or GPU depending on availability

If you use a phone photo of a scan, the app tries to improve it by:
- correcting orientation
- cropping to the scan region
- enhancing grayscale contrast
- padding to a square format
- resizing to the model input size

## Git Notes

The repository is configured to ignore local-only artifacts such as:
- `.venv/`
- `data/`
- `screenshots/`
- `static/processed_uploads/`
- `alz_model.pth`

This keeps the GitHub repo focused on source code and project files.

## Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Pillow
- NumPy

## Next Improvements

- add confidence-based warnings in the UI
- add persistent prediction history
- add drag-and-drop upload interactions
- add exportable experiment summaries
