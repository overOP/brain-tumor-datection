# Brain Tumor Detection

An end-to-end computer vision project that detects brain tumors in MRI images using **YOLOv8**, served through a **FastAPI** REST API and packaged with **Docker**.

> **Disclaimer:** This is a learning and portfolio project. It is not a medical device and must not be used for diagnosis or any clinical decision.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker](#docker)
- [Testing and Code Quality](#testing-and-code-quality)
- [Results](#results)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

The goal is to build a complete machine learning workflow, from raw data to a deployed service:

1. Explore and understand the MRI image dataset.
2. Train a YOLOv8 object detection model to locate and classify tumors.
3. Evaluate the model with standard detection metrics.
4. Expose the trained model through a FastAPI endpoint that accepts an image and returns detections.
5. Containerize the API with Docker so it runs the same everywhere.

## Dataset

**Medical Image DataSet: Brain Tumor Detection** by pkdarabi on Kaggle.

- Source: <https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection>
- Task: object detection (bounding boxes around tumors)
- Format: brain MRI images with YOLO-format label files, split into `train`, `valid`, and `test`
- Classes: defined in `data.yaml` (check the file after download for the exact class names and order)

The dataset is **not stored in this repository**. See [Getting Started](#getting-started) for how to download it.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Package manager | uv |
| Model | YOLOv8 (Ultralytics), PyTorch |
| Data and plotting | NumPy, pandas, Pillow, OpenCV, Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Containers | Docker, Docker Compose |
| Quality | pytest, httpx, Ruff, pre-commit |
| Editor | VS Code |

## Project Structure

```
brain-tumor-detection/
├── .github/workflows/ci.yml     # CI pipeline
├── .vscode/                     # Shared editor settings
├── configs/
│   ├── data.yaml                # Dataset paths and class names
│   ├── train.yaml               # Training hyperparameters
│   └── api.yaml                 # API settings
├── data/
│   ├── raw/                     # Kaggle download (not tracked by Git)
│   │   ├── train/{images,labels}/
│   │   ├── valid/{images,labels}/
│   │   ├── test/{images,labels}/
│   │   └── data.yaml
│   └── samples/                 # A few images for testing the API
├── docs/
│   ├── architecture.md
│   └── api_usage.md
├── models/                      # Trained weights (best.pt)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── reports/
│   ├── figures/                 # Plots and sample predictions
│   └── metrics/
├── runs/                        # Ultralytics training output
├── scripts/
│   ├── download_data.sh
│   ├── train.sh
│   └── run_api.sh
├── src/brain_tumor/
│   ├── config.py                # Paths and settings in one place
│   ├── api/                     # FastAPI app (main, routes, schemas, dependencies)
│   ├── data/                    # Data exploration helpers
│   ├── training/                # Training code
│   ├── evaluation/              # Evaluation code
│   ├── inference/               # Predictor that loads weights and runs detection
│   └── utils/                   # Logging and image helpers
├── tests/
│   ├── conftest.py
│   ├── test_predictor.py
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

The code lives in a `src/` layout, so it is installed as a package and imports such as `from brain_tumor.inference.predictor import ...` work the same in notebooks, tests, and Docker. The detection logic sits in `inference/`, and the API only calls it, which keeps the API thin and easy to test.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Docker (for the container step)
- A Kaggle account (for downloading the dataset)

### 1. Clone the repository

```bash
git clone https://github.com/overOP/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create the environment and install packages

```bash
uv venv env
export UV_PROJECT_ENVIRONMENT=env
uv sync
```

`UV_PROJECT_ENVIRONMENT=env` makes `uv` use the `env/` folder instead of the default `.venv/`. Add that line to your shell profile to make it permanent.

To set the packages up from scratch instead:

```bash
uv add ultralytics fastapi "uvicorn[standard]" python-multipart pydantic-settings pyyaml pillow numpy pandas matplotlib seaborn
uv add --dev pytest httpx ruff pre-commit ipykernel kaggle
```

### 3. Verify the installation

```bash
uv run python -c "
import torch, ultralytics, fastapi, cv2, pandas
print('torch      :', torch.__version__)
print('MPS ready  :', torch.backends.mps.is_available())
print('ultralytics:', ultralytics.__version__)
print('fastapi    :', fastapi.__version__)
"
```

On an Apple Silicon Mac, `MPS ready` should be `True`. On a machine with an NVIDIA GPU, check `torch.cuda.is_available()` instead.

### 4. Download the dataset

Create a Kaggle API token from your account settings and place it at `~/.kaggle/kaggle.json`, then:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

uv run kaggle datasets download \
  -d pkdarabi/medical-image-dataset-brain-tumor-detection \
  -p data/raw --unzip
```

Or download the zip manually from the Kaggle page and unzip it into `data/raw/`. Confirm that `train/`, `valid/`, `test/`, and `data.yaml` sit directly inside `data/raw/`.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` as needed (model path, device, confidence threshold, and similar settings).

## Usage

### Train the model

```bash
uv run python -m brain_tumor.training.train
```

Training settings (model size, epochs, image size, batch size, device) are read from `configs/train.yaml`. On an Apple Silicon Mac, use `device: mps` and start with the smallest model, `yolov8n`. Training output is saved to `runs/`, and the final weights should be copied to `models/best.pt`.

### Evaluate the model

```bash
uv run python -m brain_tumor.evaluation.evaluate
```

### Run the API locally

```bash
uv run uvicorn brain_tumor.api.main:app --reload --port 8000
```

Interactive documentation is then available at <http://localhost:8000/docs>.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Returns the service status |
| POST | `/predict` | Accepts an image file and returns detections |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/samples/example.jpg"
```

### Example response

```json
{
  "filename": "example.jpg",
  "detections": [
    {
      "class_name": "example_class",
      "confidence": 0.91,
      "box": { "x1": 120.5, "y1": 88.0, "x2": 260.2, "y2": 210.7 }
    }
  ]
}
```

The class names and exact response fields are defined in `src/brain_tumor/api/schemas.py`.

## Docker

### Build and run with Docker

```bash
docker build -t brain-tumor-detection .
docker run --rm -p 8000:8000 brain-tumor-detection
```

### Run with Docker Compose

```bash
docker compose up --build
```

Notes:

- The trained weights in `models/best.pt` are copied into the image, while the raw dataset and training runs are excluded through `.dockerignore`.
- Docker containers on macOS cannot use the Apple GPU (MPS), so the container runs inference on the CPU. This is fine for a single-image API. Training is done locally.

## Testing and Code Quality

```bash
uv run pytest                  # run the tests
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pre-commit install      # enable git hooks
```

The CI workflow in `.github/workflows/ci.yml` runs linting and tests on every push.

## Results

Fill this section in after training and evaluation.

| Metric | Value |
| --- | --- |
| mAP@0.5 | TBD |
| mAP@0.5:0.95 | TBD |
| Precision | TBD |
| Recall | TBD |

Sample predictions and training curves are stored in `reports/figures/`.

## Roadmap

- [ ] Set up environment and project structure
- [ ] Download and explore the dataset
- [ ] Train the baseline YOLOv8 model
- [ ] Evaluate and analyze results
- [ ] Build the FastAPI service
- [ ] Write tests
- [ ] Dockerize the API
- [ ] Set up CI
- [ ] Write up final results

## Author

**Pradeep Chaudhary (Dev)**
GitHub: [@overOP](https://github.com/overOP)

## Acknowledgements

- Dataset by pkdarabi on Kaggle
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)