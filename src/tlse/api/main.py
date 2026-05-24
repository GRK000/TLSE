from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from tlse.api.schemas import AppConfig, DatasetSummary, HealthResponse, LiveEvent, MetricsSummary, ModelInfo, PredictionRequest, PredictionResult
from tlse.api.services.runtime import AppRuntime, demo_prediction


runtime = AppRuntime()

app = FastAPI(
    title="TLSE Studio API",
    description="Local runtime API for static hand sign recognition, metrics, and demo-mode live events.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return runtime.health()


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return runtime.models.list_models()


@app.get("/api/models/current", response_model=ModelInfo | None)
def current_model() -> ModelInfo | None:
    return runtime.models.current_model()


@app.get("/api/metrics/summary", response_model=MetricsSummary)
def metrics_summary() -> MetricsSummary:
    return runtime.metrics.summary()


@app.get("/api/dataset/summary", response_model=DatasetSummary)
def dataset_summary() -> DatasetSummary:
    return runtime.dataset.summary()


@app.post("/api/predict", response_model=PredictionResult)
def predict(request: PredictionRequest) -> PredictionResult:
    try:
        if request.demo:
            return demo_prediction(request.top_k)
        return runtime.models.predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/app/config", response_model=AppConfig)
def app_config() -> AppConfig:
    return runtime.config


@app.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            event = LiveEvent(
                event="prediction",
                result=demo_prediction(),
                tracking_status="demo",
                demo=True,
                message="Camera feed unavailable. Demo mode is active.",
            )
            await websocket.send_json(event.model_dump(mode="json"))
            await asyncio.sleep(1.2)
    except WebSocketDisconnect:
        return
