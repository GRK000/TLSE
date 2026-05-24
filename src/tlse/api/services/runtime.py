from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from tlse import __version__
from tlse.api.schemas import (
    AppConfig,
    DatasetSummary,
    HealthResponse,
    MetricsSummary,
    ModelInfo,
    PredictionItem,
    PredictionRequest,
    PredictionResult,
    utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEMO_DATASET = PROJECT_ROOT / "demo" / "sample_data" / "landmarks_demo.pickle"


class ModelRegistry:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._active_artifact: dict | None = None
        self._active_path: Path | None = None

    def list_models(self) -> list[ModelInfo]:
        if not self.models_dir.exists():
            return []

        models: list[ModelInfo] = []
        for path in sorted(self.models_dir.glob("*.p")):
            models.append(self._model_info_from_path(path))
        return models

    def current_model(self) -> ModelInfo | None:
        models = self.list_models()
        return models[0] if models else None

    def predict(self, request: PredictionRequest) -> PredictionResult:
        started = time.perf_counter()
        artifact = self._load_active_artifact()
        if artifact is None:
            result = demo_prediction(request.top_k)
            result.latency_ms = elapsed_ms(started)
            result.message = "Demo prediction returned because no local model artifact is available."
            return result

        model = artifact["model"]
        class_names = list(artifact.get("class_names", []))
        features = np.asarray(request.input.landmarks, dtype=float)
        if features.ndim != 1:
            raise ValueError("Expected a flattened landmark vector.")
        if not hasattr(model, "predict_proba"):
            raise ValueError("Active model does not expose predict_proba.")

        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        order = np.argsort(probabilities)[::-1][: request.top_k]
        top = [
            PredictionItem(label=class_names[int(index)] if class_names else str(index), confidence=float(probabilities[index]))
            for index in order
        ]
        best = top[0] if top else None
        accepted = bool(best and best.confidence >= request.confidence_threshold)
        return PredictionResult(
            prediction=best.label if best and accepted else "?",
            confidence=best.confidence if best else None,
            accepted=accepted,
            top_predictions=top,
            mode="local",
            latency_ms=elapsed_ms(started),
        )

    def _model_info_from_path(self, path: Path) -> ModelInfo:
        artifact = self._safe_load_artifact(path)
        metadata = artifact.get("metadata", {}) if artifact else {}
        created_at = parse_datetime(metadata.get("created_at"))
        return ModelInfo(
            id=path.stem,
            name=path.stem.replace("_", " ").title(),
            path=str(path.relative_to(PROJECT_ROOT)),
            feature_version=metadata.get("feature_version"),
            class_names=list(artifact.get("class_names", [])) if artifact else [],
            created_at=created_at,
            is_active=self.current_path() == path,
            is_demo=path.name.startswith("demo"),
        )

    def current_path(self) -> Path | None:
        if self._active_path and self._active_path.exists():
            return self._active_path
        paths = sorted(self.models_dir.glob("*.p")) if self.models_dir.exists() else []
        return paths[0] if paths else None

    def _load_active_artifact(self) -> dict | None:
        path = self.current_path()
        if path is None:
            return None
        if self._active_path == path and self._active_artifact is not None:
            return self._active_artifact
        self._active_artifact = self._safe_load_artifact(path)
        self._active_path = path
        return self._active_artifact

    @staticmethod
    def _safe_load_artifact(path: Path) -> dict | None:
        try:
            with path.open("rb") as file:
                artifact = pickle.load(file)
        except Exception:
            return None
        return artifact if isinstance(artifact, dict) else None


class MetricsService:
    def __init__(self, reports_dir: Path = REPORTS_DIR) -> None:
        self.reports_dir = reports_dir

    def summary(self) -> MetricsSummary:
        report = self._latest_metrics()
        if report is None:
            return MetricsSummary()
        classification = report.get("classification_report", {})
        macro_avg = classification.get("macro avg", {}) if isinstance(classification, dict) else {}
        return MetricsSummary(
            accuracy=to_float(report.get("accuracy")),
            macro_f1=to_float(report.get("macro_f1") or macro_avg.get("f1-score")),
            precision_macro=to_float(macro_avg.get("precision")),
            recall_macro=to_float(macro_avg.get("recall")),
            num_classes=int(report.get("num_classes") or 0),
            num_samples=int(report.get("num_samples") or 0),
            trained_at=parse_datetime(report.get("created_at") or report.get("trained_at")),
            average_latency_ms=to_float(report.get("average_latency_ms")),
        )

    def _latest_metrics(self) -> dict | None:
        if not self.reports_dir.exists():
            return None
        candidates = sorted(self.reports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return None


class DatasetService:
    def __init__(self, demo_dataset: Path = DEMO_DATASET) -> None:
        self.demo_dataset = demo_dataset

    def summary(self) -> DatasetSummary:
        if not self.demo_dataset.exists():
            return DatasetSummary()
        try:
            with self.demo_dataset.open("rb") as file:
                dataset = pickle.load(file)
        except Exception:
            return DatasetSummary(source=str(self.demo_dataset.relative_to(PROJECT_ROOT)))
        labels = list(dataset.get("labels", []))
        class_names = [str(item) for item in dataset.get("class_names", sorted(set(labels)))]
        counts = Counter(str(label) for label in labels)
        groups = dataset.get("groups", dataset.get("group_ids", []))
        return DatasetSummary(
            classes=class_names,
            samples_per_class={name: int(counts.get(name, 0)) for name in class_names},
            sessions=len(set(groups)) if groups is not None else 0,
            users=0,
            total_samples=len(labels),
            source=str(self.demo_dataset.relative_to(PROJECT_ROOT)),
        )


class AppRuntime:
    def __init__(self) -> None:
        self.models = ModelRegistry()
        self.metrics = MetricsService()
        self.dataset = DatasetService()
        self.config = AppConfig()

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, timestamp=utc_now(), mode=self.mode())

    def mode(self) -> str:
        return "local" if self.models.current_path() else "demo"


def demo_prediction(top_k: int = 3) -> PredictionResult:
    predictions = [
        PredictionItem(label="A", confidence=0.92),
        PredictionItem(label="E", confidence=0.05),
        PredictionItem(label="L", confidence=0.03),
    ][:top_k]
    return PredictionResult(
        prediction=predictions[0].label,
        confidence=predictions[0].confidence,
        accepted=True,
        top_predictions=predictions,
        mode="demo",
        latency_ms=0.0,
        message="Demo mode uses simulated predictions.",
    )


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
