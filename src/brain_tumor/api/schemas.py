from pydantic import BaseModel


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: Box


class PredictionResponse(BaseModel):
    filename: str | None
    detections: list[Detection]