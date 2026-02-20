import time
import uuid
import logging
import joblib

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from app.schema import PatientData, PredictionResponse
from app.config import settings
from app.logging_conf import setup_logging
from app.middlewares import RequestContextMiddleware

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------
# Load model once at startup
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...", extra={"extra_fields": {"model_path": settings.model_path}})
    app.state.model = joblib.load(settings.model_path)
    logger.info("Model loaded successfully")
    yield


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(RequestContextMiddleware)


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, data: PatientData):

    request_id = request.state.request_id

    logger.info(
        "Prediction request received",
        extra={"extra_fields": {"request_id": request_id, "payload": data.dict()}}
    )

    try:
        df = pd.DataFrame([data.dict()])
        model = request.app.state.model

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0].max()

        logger.info(
            "Prediction completed",
            extra={"extra_fields": {
                "request_id": request_id,
                "prediction": int(pred),
                "probability": f'{proba*100:.2f}%'
            }}
        )

        return PredictionResponse(
            prediction=int(pred),
            probability=f'{proba*100:.2f}%'
        )

    except Exception as e:
        logger.exception(
            "Prediction failed",
            extra={"extra_fields": {"request_id": request_id}}
        )
        raise HTTPException(status_code=500, detail="Internal server error")
