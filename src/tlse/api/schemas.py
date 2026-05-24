from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


RuntimeMode = Literal["demo", "local"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    timestamp: datetime
    mode: RuntimeMode


class ModelInfo(BaseModel):
    id: str
    name: str
    path: str
    feature_version: str | None = None
    class_names: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    is_active: bool = False
    is_demo: bool = False


class MetricsSummary(BaseModel):
    accuracy: float | None = None
    macro_f1: float | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None
    num_classes: int = 0
    num_samples: int = 0
    trained_at: datetime | None = None
    average_latency_ms: float | None = None


class DatasetSummary(BaseModel):
    classes: list[str] = Field(default_factory=list)
    samples_per_class: dict[str, int] = Field(default_factory=dict)
    sessions: int = 0
    users: int = 0
    total_samples: int = 0
    source: str | None = None


class PredictionInput(BaseModel):
    landmarks: list[float] = Field(default_factory=list, description="Flattened landmark feature vector.")


class PredictionRequest(BaseModel):
    input: PredictionInput
    top_k: int = Field(default=3, ge=1, le=10)
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    demo: bool = False


class PredictionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    prediction: str | None
    confidence: float | None
    accepted: bool
    top_predictions: list[PredictionItem] = Field(default_factory=list)
    mode: RuntimeMode
    latency_ms: float
    message: str | None = None


class LiveEvent(BaseModel):
    event: Literal["prediction", "status"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: PredictionResult | None = None
    tracking_status: Literal["hand_detected", "searching", "low_confidence", "camera_error", "demo"]
    demo: bool = False
    message: str


class AppConfig(BaseModel):
    confidence_threshold: float = 0.75
    smoothing_window: int = 8
    theme_default: Literal["system", "light", "dark"] = "system"
    feature_flags: dict[str, bool] = Field(
        default_factory=lambda: {
            "live_websocket": True,
            "dataset_capture": False,
            "model_promotion": False,
            "speech": True,
            "tauri_ready": False,
        }
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
