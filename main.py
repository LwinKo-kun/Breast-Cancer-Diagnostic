from contextlib import asynccontextmanager
from typing import Any, Dict, List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.project_paths import MODEL_PATH

ml_context: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifact = joblib.load(MODEL_PATH)
        ml_context["pipeline"] = artifact["pipeline"]
        ml_context["feature_names"] = artifact["feature_names"]
        ml_context["target_names"] = artifact["target_names"]
        ml_context["best_params"] = artifact.get("best_params", {})
        print(f"Artifact loaded from {MODEL_PATH}")
    except FileNotFoundError:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Execute 'python -m src.train' first."
        )

    yield
    ml_context.clear()


app = FastAPI(
    title="Breast Cancer Diagnostic Classifier API",
    description="Inference API powered by a tuned K-Nearest Neighbors pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

BENIGN_SAMPLE = [
    13.54,
    14.36,
    87.46,
    566.3,
    0.09779,
    0.08129,
    0.06664,
    0.04781,
    0.1885,
    0.05766,
    0.2699,
    0.7886,
    2.058,
    23.56,
    0.008462,
    0.0146,
    0.02387,
    0.01315,
    0.0198,
    0.0023,
    15.11,
    19.26,
    99.7,
    711.2,
    0.144,
    0.1773,
    0.239,
    0.1288,
    0.2977,
    0.07259,
]


class DiagnosticRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Array of 30 continuous cell nuclei metrics.",
        examples=[BENIGN_SAMPLE],
    )


class DiagnosticResponse(BaseModel):
    prediction: str
    prediction_code: int
    confidence_pct: float
    class_probabilities: Dict[str, float]
    nearest_neighbors_distances: List[float]


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in ml_context,
        "features_expected": len(ml_context.get("feature_names", [])),
    }


@app.post("/predict", response_model=DiagnosticResponse)
def predict(payload: DiagnosticRequest):
    pipeline = ml_context.get("pipeline")
    target_names = ml_context.get("target_names")

    if not pipeline or not target_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized.",
        )

    input_data = np.array(payload.features, dtype=float).reshape(1, -1)

    pred_idx = int(pipeline.predict(input_data)[0])
    probabilities = pipeline.predict_proba(input_data)[0]

    scaler = pipeline.named_steps["scaler"]
    knn = pipeline.named_steps["knn"]

    scaled_data = scaler.transform(input_data)
    distances, _ = knn.kneighbors(scaled_data)

    return DiagnosticResponse(
        prediction=target_names[pred_idx],
        prediction_code=pred_idx,
        confidence_pct=round(float(probabilities[pred_idx]) * 100, 2),
        class_probabilities={
            target_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(target_names))
        },
        nearest_neighbors_distances=[round(float(d), 4) for d in distances[0]],
    )