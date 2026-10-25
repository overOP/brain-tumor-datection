# Brain Tumor Detection

An end-to-end computer vision project that detects and classifies brain tumors in MRI images using **YOLOv8**, served through a **FastAPI** REST API and packaged with **Docker**.

> **Disclaimer:** This is a learning and portfolio project. It is not a medical device and must not be used for diagnosis or any clinical decision.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker](#docker)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

The goal is to build a complete machine learning workflow, from raw data to a deployed service:

1. Explore and understand the MRI image dataset.
2. Train a YOLOv8 object detection model to locate and classify tumors.
3. Evaluate the model on a held-out test set and analyze its mistakes.
4. Expose the trained model through a FastAPI endpoint that accepts an image and returns detections.
5. Containerize the API with Docker so it runs the same everywhere.

## Dataset

**Medical Image DataSet: Brain Tumor Detection** by pkdarabi on Kaggle.

- Source: <https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection>
- Task: object detection
- Classes: `glioma`, `meningioma`, `pituitary`
- Image size: 640 x 640 for every image
- Labels: YOLO-format `.txt` files, one per image. The labels are **polygons** (tumor outlines), not plain boxes. Ultralytics converts them to bounding boxes automatically for detection training.

| Split | Images |
| --- | --- |
| Train | 2143 |
| Valid | 612 |
| Test | 308 |

Findings from the data exploration (`notebooks/01_data_exploration.ipynb`):

- Every image has exactly one tumor, and there are no empty label files.
- The class split is moderately imbalanced. In the training set glioma is about 46% of the labels, pituitary about 31%, and meningioma about 23%. Valid and test have the same proportions.
- Pituitary tumors are the smallest (median box about 1.9% of the image), and glioma sizes vary the most (0.3% to 49.9%).
- One meningioma label in the training set was a 2 px sliver along the image edge, a labelling error. It was moved out of the dataset into `data/excluded/`, which leaves 2143 training images.

The dataset is **not stored in this repository**. See [Getting Started](#getting-started) for how to download it.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.13 |
| Package manager | uv |
| Model | YOLOv8n (Ultralytics), PyTorch |
| Data and plotting | NumPy, pandas, Pillow, OpenCV, Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Containers | Docker, Docker Compose |
| Tooling | pytest, httpx, Ruff, pre-commit |
| Editor | VS Code |

## Project Structure

```
brain-tumor-detection/
├── configs/
│   ├── data.yaml                # Dataset paths and class names for YOLO
│   ├── train.yaml
│   └── api.yaml
├── data/
│   ├── raw/                     # Kaggle download (not tracked by Git)
│   ├── excluded/                # Removed bad label (not tracked)
│   └── samples/
├── models/                      # Trained weights (best.pt, not tracked)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── runs/                        # Ultralytics training output (not tracked)
├── scripts/
├── src/brain_tumor/
│   ├── config.py
│   ├── api/                     # FastAPI app: main, routes, schemas
│   ├── data/
│   ├── training/train.py        # Training script
│   ├── evaluation/evaluate.py   # Test-set evaluation
│   ├── inference/predictor.py   # Loads weights and runs detection
│   └── utils/
├── tests/
│   ├── conftest.py
│   ├── test_predictor.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

The code uses a `src/` layout, so it is installed as a package and imports such as `from brain_tumor.inference.predictor import Predictor` work the same in notebooks, scripts, and Docker. The detection logic lives in `inference/`, and the API only calls it, which keeps the API thin and easy to test.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Docker Desktop (only for the container step)
- A Kaggle account (to download the dataset)

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

`UV_PROJECT_ENVIRONMENT=env` makes `uv` use the `env/` folder instead of the default `.venv/`. Run the `export` line in every new terminal, or use `uv run --active` after activating `env`.

### 3. Verify the installation

```bash
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

On an Apple Silicon Mac this should print `True` for MPS. On a machine with an NVIDIA GPU, check `torch.cuda.is_available()` instead.

### 4. Download the dataset

Create a Kaggle API token from your account settings and place it at `~/.kaggle/kaggle.json`, then:

```bash
uv run kaggle datasets download \
  -d pkdarabi/medical-image-dataset-brain-tumor-detection \
  -p data/raw --unzip
```

You can also download the zip manually and unzip it into `data/raw/`. Make sure `train/`, `valid/`, `test/`, and `data.yaml` sit directly inside `data/raw/`.

### 5. Point the training config at your data

Edit `configs/data.yaml` so `path` is the absolute path to `data/raw` on your machine:

```yaml
path: /absolute/path/to/brain-tumor-detection/data/raw
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['glioma', 'meningioma', 'pituitary']
```

## Usage

### Train the model

```bash
uv run python -m brain_tumor.training.train --epochs 30 --name yolov8n_baseline
```

Settings used for the baseline: `yolov8n.pt` pretrained weights, image size 640, batch size 8, device `mps`, seed 42. Output is saved to `runs/`. Copy the final weights to `models/best.pt`:

```bash
cp runs/<your-run-name>/weights/best.pt models/best.pt
```

### Evaluate on the test set

```bash
uv run python -m brain_tumor.evaluation.evaluate
```

### Run the API locally

```bash
uv run uvicorn brain_tumor.api.main:app --reload --port 8000
```

Interactive documentation is then available at <http://localhost:8000/docs>.

## Results

Model: YOLOv8n, 30 epochs (about 1.3 hours on an Apple M1), trained at 640 x 640.

**Test set (308 images, not used for training or model selection):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| **All** | **0.903** | **0.822** | **0.907** | **0.671** |
| glioma | 0.849 | 0.635 | 0.789 | 0.509 |
| meningioma | 0.906 | 0.936 | 0.965 | 0.790 |
| pituitary | 0.955 | 0.897 | 0.967 | 0.714 |

For comparison, the validation set (612 images, used to pick the best epoch) gave mAP50 0.913 and mAP50-95 0.698.

Takeaways:

- The model generalizes well. Test scores are close to validation scores.
- **Glioma is the weakest class**, with a recall of 0.635. Glioma also has the most varied size and shape in the data.
- Meningioma and pituitary both reach a recall near or above 0.9.
- Inference takes about 3 ms per image on the M1 GPU.

Training curves and the confusion matrix are saved in the `runs/` folder of each run.

## Error Analysis

The full analysis is in `notebooks/02_results_analysis.ipynb`. For each of the 308 test images, the single best detection was compared with the true tumor outline. A detection counts as correct when the class matches and the box overlaps the true box by at least 50% (IoU). This check is stricter than mAP, so its counts differ from the table above.

| True class | Correct | Bad box | Missed | Wrong class |
| --- | --- | --- | --- | --- |
| glioma (159) | 113 | 26 | 17 | 3 |
| meningioma (62) | 59 | 2 | 1 | 0 |
| pituitary (87) | 80 | 1 | 3 | 3 |

What it shows:

- Glioma is rarely confused with the other tumor types (3 of 159). Its weakness is localization and detection, not classification.
- Most of the 26 glioma bad boxes are large, irregular outlines where the model boxes only part of the tumor. A few are confident boxes on dark regions that are not the labelled tumor.
- Of the 17 missed gliomas, rerunning at a confidence threshold of 0.05 recovered 6, which means the model had found them with low confidence. Another 8 had only boxes elsewhere or partly on the tumor, and 3 had no detection at all.
- The images with no detection looked low-contrast to the eye. This is a visual impression from a small sample, not a measured result.
- Lowering the confidence threshold could recover some misses, but its effect on false detections has not been measured.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Returns the service status |
| POST | `/predict` | Accepts an image file and returns detections |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/raw/test/images/<image-name>.jpg"
```

### Example response

```json
{
  "filename": "example.jpg",
  "detections": [
    {
      "class_name": "glioma",
      "confidence": 0.62,
      "box": { "x1": 166.8, "y1": 336.2, "x2": 346.4, "y2": 478.1 }
    }
  ]
}
```

Boxes are in pixel coordinates of the uploaded image. A file that is not a valid image returns HTTP 400. Detection uses a confidence threshold of 0.25 and a non-max suppression IoU of 0.5.

## Testing

```bash
uv run pytest -v
```

Six tests cover the predictor (output format, invalid input) and the API (`/health`, a valid upload, and rejection of a non-image file). They use a generated gray image, so they check that the code runs and returns the right format, not the model's accuracy. Model accuracy is measured by the evaluation script. The tests are skipped when `models/best.pt` is not present.

## Docker

The image packages the API, its dependencies, and the trained weights.

> **Weights are not in Git.** `*.pt` files are ignored, so after cloning the repository you must place `best.pt` in `models/` before building the image. Otherwise the build stops at the `COPY models/best.pt` step.

```bash
docker compose up --build
```

Then test it:

```bash
curl -s localhost:8000/health
```

Notes:

- On Linux, `pyproject.toml` takes PyTorch from the CPU-only index, which keeps the image and the download small. macOS installs keep the normal build with Apple GPU support.
- Docker containers on macOS cannot use the Apple GPU, so the container runs inference on the CPU. This is fine for single-image requests. Training is done locally.
- The `.dockerignore` file keeps the dataset, training runs, and notebooks out of the image.

## Limitations

- Trained and tested on a single public dataset. It was not validated on data from other scanners or hospitals, so real-world performance is unknown.
- Every image in the dataset contains exactly one tumor, and there is no "no tumor" class. The model has not been trained to say that an image is healthy, so it should not be used that way.
- Glioma recall is low (0.635 on the test set). The error analysis shows that glioma is rarely confused with other classes. Most failures are loosely placed boxes on irregular tumors, low-confidence detections, and a few tumors that are not detected at all.
- Detection only. The polygon labels would also allow a segmentation model, which is not part of this project yet.

## Roadmap

- [x] Set up the environment and project structure
- [x] Download and explore the dataset
- [x] Train the baseline YOLOv8n model
- [x] Evaluate on the test set
- [x] Build the FastAPI service
- [x] Dockerize the API
- [x] Write automated tests
- [x] Error analysis for glioma misses
- [ ] Add a Makefile and CI workflow
- [ ] Measure the effect of the confidence threshold on precision and recall
- [ ] Try a larger model or longer training to improve glioma recall

## Author

**Pradeep Chaudhary (Dev)**
GitHub: [@overOP](https://github.com/overOP)

## Acknowledgements

- Dataset by pkdarabi on Kaggle
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
