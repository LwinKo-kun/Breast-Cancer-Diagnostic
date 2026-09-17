import random
import time
import pytest
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

API_URL = "http://127.0.0.1:8000"


def test_health_check():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            "FastAPI server is offline. Run 'uvicorn main:app --reload' to run live API tests."
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features_expected"] == 30


def test_live_prediction_accuracy():
    try:
        requests.get(f"{API_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip("FastAPI server offline.")

    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    random.seed(99)
    test_indices = random.sample(range(len(X_test)), 7)

    matches = 0
    for idx in test_indices:
        res = requests.post(
            f"{API_URL}/predict", json={"features": X_test[idx].tolist()}
        )
        assert res.status_code == 200
        payload = res.json()
        assert "prediction" in payload
        assert "confidence_pct" in payload
        assert len(payload["nearest_neighbors_distances"]) > 0

        expected = data.target_names[y_test[idx]]
        if payload["prediction"].lower() == expected.lower():
            matches += 1

    assert (
        matches / len(test_indices)
    ) >= 0.70, "Batch accuracy on holdout test cases dropped below threshold."