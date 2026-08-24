"""HTTP API для предсказания отказа / HTTP API for failure prediction."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.predict import load_model, predict
from src.schemas import PredictionResponse, SensorReading

ARTIFACT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ARTIFACT
    ARTIFACT = load_model()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Бинарная классификация отказа по показаниям датчиков / Binary failure classification from sensors.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Корень сайта ведёт в Swagger / Root URL redirects to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Проверка, что сервис и модель живы / Liveness check for service and model."""
    return {"status": "ok", "model_loaded": ARTIFACT is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(reading: SensorReading):
    """Принимает JSON с датчиками, возвращает класс и вероятность отказа."""
    try:
        pred, prob = predict(reading.to_raw_dict(), artifact=ARTIFACT)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    label = "failure" if int(pred) == 1 else "no_failure"
    return PredictionResponse(prediction=int(pred), probability=float(prob), label=label)
