from fastapi.testclient import TestClient

from tlse.api.main import app
from tlse.api.schemas import PredictionRequest


client = TestClient(app)


def test_health_endpoint_returns_runtime_status():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["mode"] in {"demo", "local"}
    assert payload["timestamp"]


def test_metrics_endpoint_handles_empty_or_local_reports():
    response = client.get("/api/metrics/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "accuracy" in payload
    assert "macro_f1" in payload
    assert "num_classes" in payload


def test_prediction_schema_accepts_landmark_vector():
    request = PredictionRequest(input={"landmarks": [0.0] * 47}, top_k=3)

    assert len(request.input.landmarks) == 47
    assert request.confidence_threshold == 0.75
