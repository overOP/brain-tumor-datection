from contextlib import asynccontextmanager

from fastapi import FastAPI

from brain_tumor.api.routes import router
from brain_tumor.inference.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Brain Tumor Detection API", lifespan=lifespan)
app.include_router(router)