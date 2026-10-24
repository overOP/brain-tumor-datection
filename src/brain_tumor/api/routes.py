from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import UnidentifiedImageError

from brain_tumor.api.schemas import PredictionResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    try:
        detections = request.app.state.predictor.predict(data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return {"filename": file.filename, "detections": detections}