from fastapi.testclient import TestClient

from src.api.main import app, model

client = TestClient(app)

SAMPLE_INPUT = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1200.5,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 180,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Bank Marketing Prediction API is running"


def test_health_reports_model_state():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_predict_rejects_invalid_input():
    """Schema validation should reject a missing field whether or not a model is loaded."""
    bad_input = {k: v for k, v in SAMPLE_INPUT.items() if k != "age"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_returns_prediction_or_503():
    response = client.post("/predict", json=SAMPLE_INPUT)

    if model is None:
        assert response.status_code == 503
        return

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["yes", "no"]
    assert 0.0 <= data["probability_yes"] <= 1.0
