# MIRA Production Guide

## Purpose

This document records the current architecture, production-readiness review, deployment guidance, and operational notes for the MIRA Alzheimer's MRI research application.

MIRA is a local Flask and PyTorch research app. It is suitable for demos, coursework, and controlled internal experimentation. It should not be treated as a clinical production system.

## Documentation Format Decision

The primary documentation format should be Markdown because it is version-controlled, easy to review in Git, and simple to keep synchronized with code changes.

A PDF is still useful for this project because:
- it is easier to share with advisers, evaluators, or non-technical reviewers
- it creates a stable handoff artifact for checkpoints or presentations
- it can be linked directly from the About page for quick access

For that reason this project now keeps:
- a source document in `docs/MIRA-Production-Guide.md`
- an exported PDF in `static/docs/mira-production-guide.pdf`

## Code Review Summary

### Strengths already present

- Clear route-level separation for home, upload, analytics, about, install, and prediction flows
- Consistent model class mapping through saved checkpoint metadata
- A user-facing upload and result flow that is easy to follow
- Research artifact surfacing through charts, confusion matrices, and Grad-CAM outputs
- Training code that is readable and reasonably modular for a coursework project

### Gaps found before the update

- A hard-coded Flask secret key was checked into source
- The app had no request-size limit for uploaded files
- Broad exception handling surfaced raw exception strings to users
- Model loading depended on `__main__`, which is fragile for imports or alternate runners
- Some UI and README text described preprocessing steps that the current inference code does not actually perform
- The About page did not expose project documentation in a durable way
- Static caching logic in the service worker was too permissive and cached any fetched response

### Improvements implemented in this update

- Moved secret-key handling to environment-based configuration with a generated fallback
- Added `MAX_CONTENT_LENGTH` to cap upload size
- Added safer session-cookie defaults for local HTTPS usage
- Added security-oriented response headers for content type, framing, and referrer behavior
- Refactored model loading behind a cached accessor so the app can import cleanly and still resolve the model when prediction is requested
- Added structured logging for startup, model loading, and predictions
- Replaced raw unexpected exception exposure with server-side logging and a generic user-facing error message
- Added a `413` handler for oversized uploads
- Updated UI copy so it matches the actual inference pipeline
- Added maintainable documentation plus a generated PDF linked from the About page

## Current Architecture

### Application components

- `app.py`: Flask entry point, request handling, inference preprocessing, and static routes
- `train.py`: EfficientNet-B0 training loop, dashboards, confusion matrix generation, Grad-CAM, and markdown report exports
- `templates/`: Multi-page UI templates
- `static/`: CSS, PWA assets, icons, service worker, and generated public documentation assets
- `screenshots/`: Training dashboards and exported research images
- `data/`: Class-folder image dataset used for training

### Main web routes

- `/`: Home page
- `/upload`: MRI upload page
- `/analytics`: Training visualization page
- `/about`: Project notes and documentation links
- `/install`: Local HTTPS and install guide
- `/predict`: POST inference endpoint
- `/downloads/local-ca`: Local certificate download helper
- `/artifacts/<path>`: Training artifact file serving
- `/manifest.webmanifest`: PWA manifest
- `/service-worker.js`: Service worker
- `/assets/<path>`: Project asset file serving

## Inference Pipeline Notes

The current production-aligned description of inference is:

1. Validate that an uploaded file is present and has an allowed extension.
2. Save the upload to a temporary file.
3. Normalize orientation with EXIF transpose handling.
4. Save a reviewable preview image for the result page.
5. Resize the inference image directly to `224 x 224` to match the training pipeline.
6. Convert to tensor and apply ImageNet normalization.
7. Run EfficientNet-B0 inference.
8. Return the predicted label, confidence score, and saved preview.

Important clarification:
The current inference path does not crop, enhance, sharpen, or pad the scan before prediction. Earlier project text suggested those steps, but that did not match the live code. The updated UI and documentation now reflect the real behavior.

## Configuration

### Environment variables

- `MIRA_HOST`: Host bind address. Default `0.0.0.0`.
- `MIRA_PORT`: Port. Default `5000`.
- `MIRA_USE_HTTPS`: Enable HTTPS when certificates exist. Default `1`.
- `MIRA_DEBUG`: Enable Flask debug mode. Default `0`.
- `MIRA_LOCAL_CA_PATH`: Optional explicit path to a local CA certificate.
- `MIRA_SECRET_KEY`: Recommended explicit Flask secret key for stable sessions.
- `MIRA_MAX_UPLOAD_MB`: Maximum upload size in megabytes. Default `10`.

### HTTPS behavior

If `certs/mira-local.pem` and `certs/mira-local-key.pem` exist and HTTPS is enabled, the app uses them.
If HTTPS is enabled and those files are missing, Flask falls back to an ad hoc certificate.

## Production-Level Guidance

### What is now reasonably in place

- safer configuration defaults
- request size limiting
- less sensitive error exposure
- basic security headers
- clearer operational logging
- documentation linked from the UI
- better import behavior for alternate startup patterns

### What would still be required for a real production healthcare deployment

- authentication and authorization
- encrypted storage and retention rules for uploaded images
- audit logs and traceability controls
- reverse proxy and production WSGI or ASGI serving stack
- monitoring, alerting, and health checks
- model versioning and rollback support
- stricter validation and test coverage
- privacy, consent, and regulatory compliance processes
- formal clinical evaluation and human oversight workflows

## Training Script Notes

`train.py` is reasonably structured for a research pipeline. It already includes:
- deterministic seed setup
- explicit class validation
- modular plotting helpers
- confusion-matrix and Grad-CAM exports
- checkpoint saving with class-name metadata

Main remaining concerns for `train.py` are operational rather than stylistic:
- no command-line interface or config file for tunable parameters
- no automated tests for data assumptions or output artifacts
- long training runs rely on local interactive plotting
- output retention and run versioning are not formalized

## File References

- Source documentation: `docs/MIRA-Production-Guide.md`
- Shareable PDF: `static/docs/mira-production-guide.pdf`
- Flask app entry point: `app.py`
- Training pipeline: `train.py`

## Recommendation

For this project, keep Markdown as the authoritative documentation format and keep the PDF as a generated companion artifact.

That combination gives you:
- maintainability for developers
- a clean handoff document for reviewers
- a direct About-page link inside the app
