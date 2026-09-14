import joblib
import numpy as np
import pandas as pd

from project_paths import MODEL_PATH, get_data_path

# 1. Load training data to establish distribution baselines and verify non-existence
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
scaler = pipeline.named_steps["scaler"]
knn = pipeline.named_steps["knn"]

# Revert scaled coordinates back to real-world measurements
X_trained = scaler.inverse_transform(knn._fit_X)

# 2. Configuration for synthetic data generation
np.random.seed(101)
num_samples_to_generate = 10  # Set the number of rows you want
perturbation_ratio = 0.08     # 8% standard deviation jitter
min_separation_dist = 0.05    # Minimum Euclidean distance from any training point

std_vector = np.std(X_trained, axis=0)
novel_samples = []

# 3. Generate candidate vectors and check against existing points
while len(novel_samples) < num_samples_to_generate:
    # Pick an existing reference instance and perturb it
    base_sample = X_trained[np.random.randint(0, len(X_trained))]
    noise = np.random.normal(0, perturbation_ratio * std_vector, size=base_sample.shape)
    candidate = np.clip(base_sample + noise, a_min=0, a_max=None)

    # Ensure the new sample is strictly novel (does not exist in the training set)
    distances = np.linalg.norm(X_trained - candidate, axis=1)
    if np.min(distances) > min_separation_dist:
        novel_samples.append(candidate)

# 4. Convert to DataFrame, round, and export
df_novel = pd.DataFrame(novel_samples, columns=feature_names)
df_novel = df_novel.round(4)

csv_path = get_data_path("unlabeled_synthetic_data.csv")
df_novel.to_csv(csv_path, index=False)

print(f"Successfully created {num_samples_to_generate} unique, unseen feature records.")
print(f"Exported to '{csv_path}' with {len(feature_names)} features and 0 predictions.\n")
print("Preview (First 5 records, first 5 columns):")
print(df_novel.iloc[:5, :5])