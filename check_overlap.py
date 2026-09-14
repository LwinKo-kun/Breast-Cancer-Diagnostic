import sys

import joblib
import numpy as np
import pandas as pd

from project_paths import MODEL_PATH

# 1. Target file to check
TARGET_CSV = "unlabeled_synthetic_data.csv"
if len(sys.argv) > 1:
    TARGET_CSV = sys.argv[1]

# 2. Extract memorized training instances from the model bundle
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
scaler = pipeline.named_steps["scaler"]
knn = pipeline.named_steps["knn"]

# Revert to unscaled clinical coordinates
X_train = scaler.inverse_transform(knn._fit_X)
feature_names = artifact["feature_names"]

print(f"Loaded {len(X_train)} memorized reference points from model.joblib.")

# 3. Load user input data
try:
    df_input = pd.read_csv(TARGET_CSV)
except FileNotFoundError:
    print(f"Error: File '{TARGET_CSV}' not found.")
    sys.exit(1)

# Ensure only feature columns are evaluated (ignore metadata/labels if present)
input_cols = [col for col in feature_names if col in df_input.columns]
if len(input_cols) != len(feature_names):
    print(f"Error: Target file missing required feature columns. Expected 30, got {len(input_cols)}.")
    sys.exit(1)

X_input = df_input[input_cols].values
print(f"Checking {len(X_input)} records from '{TARGET_CSV}' against training data...\n")

# 4. Numerical collision and proximity checks
tolerance = 1e-5
collisions = 0
results = []

for idx, candidate in enumerate(X_input):
    # Compute Euclidean distance from candidate to all training points
    distances = np.linalg.norm(X_train - candidate, axis=1)
    min_distance = np.min(distances)
    closest_train_idx = np.argmin(distances)
    
    is_overlap = min_distance < tolerance
    if is_overlap:
        collisions += 1

    status = "🚨 OVERLAP (Duplicate)" if is_overlap else "✅ Unique"
    results.append({
        "Row": idx + 1,
        "Status": status,
        "Min Euclidean Distance": round(float(min_distance), 4),
        "Nearest Training Index": int(closest_train_idx)
    })

# 5. Display tabular summary
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

print("\n--- Final Assessment ---")
if collisions == 0:
    print(f"Safe to proceed: 0 duplicates detected. All {len(X_input)} rows are strictly unique.")
    print(f"Closest overall point distance: {summary_df['Min Euclidean Distance'].min():.4f}")
else:
    print(f"Warning: {collisions} record(s) directly overlap with memorized training data.")