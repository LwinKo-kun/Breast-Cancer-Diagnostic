import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load memorized training instances from the model artifact
artifact = joblib.load("model.joblib")
pipeline = artifact["pipeline"]
scaler = pipeline.named_steps["scaler"]
knn = pipeline.named_steps["knn"]

# Revert to original clinical units for direct numeric comparison
X_trained_unscaled = scaler.inverse_transform(knn._fit_X)
print(f"Loaded {len(X_trained_unscaled)} memorized training points from model.joblib.")

# 2. Re-create the holdout partition
data = load_breast_cancer()
_, X_test_unseen, _, _ = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
print(f"Loaded {len(X_test_unseen)} samples from the unseen holdout partition.\n")

# 3. Check for exact numerical duplicates or near-zero Euclidean distance
identical_matches = 0
tolerance = 1e-5

for idx, test_sample in enumerate(X_test_unseen):
    distances = np.linalg.norm(X_trained_unscaled - test_sample, axis=1)
    min_dist = np.min(distances)
    
    if min_dist < tolerance:
        identical_matches += 1
        print(f"⚠️ Leak detected! Holdout Sample #{idx} matches an existing training point (distance={min_dist:.6f}).")

if identical_matches == 0:
    print("✅ Zero overlap found: 100% of the test partition is distinct from the memorized training set.")
else:
    print(f"Found {identical_matches} identical samples across partitions.")