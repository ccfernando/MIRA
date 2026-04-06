# MIRA

Medical Image Recognition for Alzheimer's

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-0b7285.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet%20Pipeline-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-12b5cb.svg)](#)

MIRA is a Flask + PyTorch web application for classifying Alzheimer's disease stages from MRI images using an EfficientNet-B0 model. It combines model training, MRI upload inference, phone-photo preprocessing, PWA install support, and research outputs such as training dashboards, confusion matrices, Grad-CAM visualizations, and markdown reports.

This project is intended for research and education only.

## Overview

MIRA supports two main workflows:

1. Training a multi-class Alzheimer's stage classifier from MRI image folders in `data/`
2. Running a local HTTPS web app that lets you upload MRI images, review predictions, and install MIRA as a phone web app on the same Wi-Fi network

The app uses separate pages instead of a single scrolling dashboard:

- `Home` for the overview and system summary
- `Upload` for MRI submission
- `Results` for saved training visualizations
- `About` for project context
- `Install` for local HTTPS trust and phone web app setup
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

### Web App and PWA

- Flask-based multi-page web UI
- Dedicated pages for Home, Upload, Results, About, Install, and Prediction Result
- Sticky top navigation on desktop and mobile-friendly bottom navigation
- Browser-aware `Install App` flow for iPhone Safari, iPhone Chrome, Android Chrome, and fallback browsers
- Local certificate download route for phone trust setup
- Manifest, service worker, and standalone install metadata for Home Screen launch
- Separate app icons for browser tab favicon vs installed phone web app icon

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

Python packages are listed in `requirements.txt`.

Install them with:

```powershell
python -m pip install -r requirements.txt
```

Current Python dependencies:

- `Flask`
- `matplotlib`
- `numpy`
- `Pillow`
- `scikit-learn`
- `torch`
- `torchvision`
- `cryptography`

Additional local tooling for HTTPS:

- `mkcert` is recommended for generating locally trusted development certificates
- a root CA file such as `rootCA.pem` must be available on the host machine for the phone trust download route

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

By default the app starts with HTTPS enabled if `certs/mira-local.pem` and `certs/mira-local-key.pem` exist.

Expected local URLs:

```text
https://127.0.0.1:5000
https://<your-lan-ip>:5000
```

Optional environment variables:

- `MIRA_HOST` to override the bind host
- `MIRA_PORT` to override the port
- `MIRA_USE_HTTPS=0` to disable HTTPS
- `MIRA_DEBUG=1` to enable Flask debug mode
- `MIRA_LOCAL_CA_PATH` to point to a specific `rootCA.pem`

## HTTPS Setup

MIRA now supports local phone installation as a web app over trusted HTTPS on the same Wi-Fi network.

### Certificate files used by the app

MIRA looks for these certificate files when it starts:

- `certs/mira-local.pem`
- `certs/mira-local-key.pem`

If both files exist and `MIRA_USE_HTTPS` is enabled, Flask serves the app over HTTPS.

### How to generate local HTTPS certificates

Recommended workflow with `mkcert` on Windows:

```powershell
mkcert -install
mkcert -key-file certs/mira-local-key.pem -cert-file certs/mira-local.pem localhost 127.0.0.1 ::1 10.10.11.13
```

Replace `10.10.11.13` with your current LAN IP if it changes.

### Local CA file used for phone trust setup

The `/downloads/local-ca` route looks for a local CA file in this order:

1. `MIRA_LOCAL_CA_PATH`
2. `%USERPROFILE%\AppData\Local\mkcert\rootCA.pem`
3. `./rootCA.pem`
4. `certs/rootCA.pem`

If you generated certificates with `mkcert`, the default path is usually enough.

## Phone Web App Install Flow

### What was added to make phone install work

The app now includes:

- a dedicated install/setup page at `/install`
- a downloadable local CA route at `/downloads/local-ca`
- a browser-aware `Install App` button in the top menu
- versioned PWA assets so old service worker caches are refreshed
- a dedicated Apple touch icon for iPhone Home Screen installs
- a separate black favicon for the browser tab

### iPhone Safari

1. Start MIRA and note the HTTPS LAN URL.
2. On the iPhone, open `https://<your-lan-ip>:5000/install` in Safari.
3. Tap `Download Local CA`.
4. Install the downloaded profile on the iPhone.
5. Go to `Settings > General > About > Certificate Trust Settings` and enable full trust for the installed local CA.
6. Reopen the MIRA HTTPS URL in Safari.
7. Tap `Install App` in MIRA or use `Share > Add to Home Screen`.
8. Remove any older MIRA shortcut first if it only showed a letter icon.

### iPhone Chrome

- Chrome on iPhone can add shortcuts, but Safari is the recommended path for a true Home Screen web app with better icon and standalone behavior.
- The MIRA install flow now redirects iPhone Chrome users to the install guide instead of showing a misleading native prompt message.

### Android Chrome

1. Open the HTTPS MIRA URL on the same Wi-Fi network.
2. Visit `/install` if certificate trust or install guidance is needed.
3. Tap `Install App` in the top menu.
4. If Chrome does not show the native prompt yet, use the Chrome menu and choose `Install app` or `Add to Home screen`.

## Web Routes

The Flask app currently exposes these main routes:

- `/` -> Home page
- `/upload` -> MRI upload page
- `/analytics` -> saved training visualizations page
- `/about` -> project summary page
- `/install` -> phone install and HTTPS trust guide
- `/downloads/local-ca` -> download the local CA file for phone trust setup
- `/predict` -> POST route for inference
- `/artifacts/<path:filename>` -> training image artifacts
- `/assets/<path:filename>` -> custom assets such as the logo
- `/manifest.webmanifest` -> web app manifest
- `/service-worker.js` -> service worker

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
|       |-- mira.png
|       `-- mira-favicon-black.png
|-- certs/
|   |-- mira-local.pem
|   `-- mira-local-key.pem
|-- templates/
|   |-- index.html
|   |-- upload.html
|   |-- analytics.html
|   |-- about.html
|   |-- install.html
|   `-- result.html
|-- static/
|   |-- manifest.webmanifest
|   |-- pwa.js
|   |-- service-worker.js
|   |-- style.css
|   |-- icons/
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
- local development certificates in `certs/`

Review `.gitignore` before pushing if you want to keep GitHub focused on source code and documentation.

## Limitations

- This is not a clinical diagnostic tool
- Prediction quality depends on data quality and model training quality
- iPhone install behavior still depends on trusting the local CA and using Safari for the best result
- local LAN installability depends on your phone trusting the HTTPS certificate chain
- results should be treated as research outputs, not medical decisions

## Roadmap Ideas

- persist upload and prediction history
- add downloadable prediction summaries
- improve analytics browsing for multiple runs
- add model/version metadata to the UI
- refine responsive behavior further
