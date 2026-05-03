from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert data["message"] == "Bank Marketing Prediction API is running"


def test_health():
    response = client.get("/health")
    assert response.json()

    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict():
    sample_input = {
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

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability_yes" in data

    assert data["prediction"] in ["yes", "no"]
    assert isinstance(data["probability_yes"], float)
    assert 0.0 <= data["probability_yes"] <= 1.0
