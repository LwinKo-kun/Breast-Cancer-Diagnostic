import joblib
import numpy as np
import pandas as pd
from src.project_paths import MODEL_PATH, get_data_path


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python -m src.train' first."
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    scaler = pipeline.named_steps["scaler"]
    knn = pipeline.named_steps["knn"]

    # Revert scaled coordinates to physical diagnostic values
    X_trained = scaler.inverse_transform(knn._fit_X)

    np.random.seed(101)
    num_samples = 10
    perturbation_ratio = 0.08
    min_separation_dist = 0.05

    std_vector = np.std(X_trained, axis=0)
    novel_samples = []

    while len(novel_samples) < num_samples:
        base_sample = X_trained[np.random.randint(0, len(X_trained))]
        noise = np.random.normal(
            0, perturbation_ratio * std_vector, size=base_sample.shape
        )
        candidate = np.clip(base_sample + noise, a_min=0, a_max=None)

        distances = np.linalg.norm(X_trained - candidate, axis=1)
        if np.min(distances) > min_separation_dist:
            novel_samples.append(candidate)

    df_novel = pd.DataFrame(novel_samples, columns=feature_names).round(4)
    csv_path = get_data_path("unlabeled_synthetic_data.csv")
    df_novel.to_csv(csv_path, index=False)

    print(f"Generated {num_samples} distinct synthetic feature vectors.")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()