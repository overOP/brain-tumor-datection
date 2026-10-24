from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]


def main():
    model = YOLO(str(ROOT / "models" / "best.pt"))
    metrics = model.val(
        data=str(ROOT / "configs" / "data.yaml"),
        split="test",
        imgsz=640,
        device="mps",
        project=str(ROOT / "runs"),
        name="test_evaluation",
    )
    print(f"Test mAP50:    {metrics.box.map50:.3f}")
    print(f"Test mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision:     {metrics.box.mp:.3f}")
    print(f"Recall:        {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()