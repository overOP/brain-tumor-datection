import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="yolov8n_baseline")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(ROOT / "configs" / "data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device="mps",
        workers=2,
        project=str(ROOT / "runs"),
        name=args.name,
        seed=42,
    )


if __name__ == "__main__":
    main()