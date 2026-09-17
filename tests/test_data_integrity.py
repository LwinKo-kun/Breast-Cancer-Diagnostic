import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.project_paths import MODEL_PATH, get_data_path


@pytest.fixture(scope="module")
def model_context():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    scaler = pipeline.named_steps["scaler"]
    knn = pipeline.named_steps["knn"]
    X_train_unscaled = scaler.inverse_transform(knn._fit_X)
    return {
        "artifact": artifact,
        "X_train": X_train_unscaled,
        "y_train": knn._y,
        "feature_names": artifact["feature_names"],
        "target_names": artifact["target_names"],
    }


def test_no_data_leakage_in_test_set(model_context):
    """Ensures test samples were not accidentally memorized in the training set."""
    data = load_breast_cancer()
    _, X_test, _, _ = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    tolerance = 1e-5
    for test_pt in X_test:
        distances = np.linalg.norm(model_context["X_train"] - test_pt, axis=1)
        assert (
            np.min(distances) > tolerance
        ), "Data leakage detected: test sample matches training sample."


def test_synthetic_data_uniqueness(model_context):
    """Verifies generated synthetic data maintains separation from training data."""
    synth_path = get_data_path("unlabeled_synthetic_data.csv")
    if not synth_path.exists():
        pytest.skip("Synthetic data not yet generated.")

    df = pd.read_csv(synth_path)
    tolerance = 1e-5
    for _, row in df.iterrows():
        vec = row[model_context["feature_names"]].values
        distances = np.linalg.norm(model_context["X_train"] - vec, axis=1)
        assert (
            np.min(distances) > tolerance
        ), "Synthetic row collided with memorized training point."


def test_export_full_trained_data(model_context):
    """Replicates inspect_data.py by exporting a clean reference CSV."""
    df = pd.DataFrame(
        model_context["X_train"], columns=model_context["feature_names"]
    )
    df["diagnosis"] = [
        model_context["target_names"][i] for i in model_context["y_train"]
    ]
    out_path = get_data_path("full_trained_data.csv")
    df.to_csv(out_path, index=False)
    assert out_path.exists()
    assert len(df) == len(model_context["y_train"])