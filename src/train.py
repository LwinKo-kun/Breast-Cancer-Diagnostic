import sys
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.project_paths import MODEL_PATH


def get_requested_sample_size(max_available: int) -> int:
    if len(sys.argv) > 1:
        try:
            val = int(sys.argv[1])
            if 10 <= val <= max_available:
                return val
        except ValueError:
            pass

    print(f"Total instances available: {max_available}")
    while True:
        try:
            choice = input(
                f"Enter number of training samples (10 to {max_available - 50}): "
            ).strip()
            val = int(choice)
            if 10 <= val <= (max_available - 50):
                return val
            print(
                f"Please choose an integer between 10 and {max_available - 50}."
            )
        except ValueError:
            print("Invalid input. Enter an integer.")


def main():
    print("[1/4] Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names.tolist()
    target_names = data.target_names.tolist()

    train_count = get_requested_sample_size(len(X))
    test_count = min(50, len(X) - train_count)

    print(
        f"\n[2/4] Slicing {train_count} training samples ({test_count} test samples)..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_count,
        test_size=test_count,
        random_state=42,
        stratify=y,
    )

    print(f"      Selected Training Set: {X_train.shape[0]} rows")
    print(
        f"      Malignant: {np.sum(y_train == 0)} | Benign: {np.sum(y_train == 1)}"
    )

    max_k = min(15, max(3, train_count // 3))
    k_candidates = [k for k in [3, 5, 7, 9, 11] if k <= max_k] or [3]

    print(f"\n[3/4] Fitting KNN Pipeline (Testing k in {k_candidates})...")
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "knn__n_neighbors": k_candidates,
            "knn__metric": ["euclidean", "manhattan"],
            "knn__weights": ["uniform", "distance"],
        },
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    print(f"      Best Parameters: {grid_search.best_params_}")
    print(f"      Cross-Val Macro F1: {grid_search.best_score_:.4f}")

    print("\n[4/4] Evaluating on holdout test set...")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    artifact = {
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "target_names": target_names,
        "best_params": grid_search.best_params_,
        "train_samples_count": int(X_train.shape[0]),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(
        f"\nSuccess: Exported model with {train_count} instances to {MODEL_PATH}."
    )


if __name__ == "__main__":
    main()