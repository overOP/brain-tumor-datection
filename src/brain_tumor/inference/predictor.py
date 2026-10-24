import io
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"


class Predictor:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cpu", conf=0.25):
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf

    def predict(self, image_bytes: bytes) -> list[dict]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = self.model.predict(
            image, imgsz=640, conf=self.conf, iou=0.5, device=self.device, verbose=False
        )[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_name": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "box": {
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                },
            })
        return detections