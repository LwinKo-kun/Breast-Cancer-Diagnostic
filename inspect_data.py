import joblib
import pandas as pd

# 1. Load the artifact
artifact = joblib.load("model.joblib")
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

# 2. Extract the scaler and KNN model from the pipeline
scaler = pipeline.named_steps["scaler"]
knn = pipeline.named_steps["knn"]

X_train_stored = knn._fit_X
y_train_stored = knn._y

# 3. Invert the standardization to view real-world clinical measurements
X_unscaled = scaler.inverse_transform(X_train_stored)

df = pd.DataFrame(X_unscaled, columns=feature_names)
df['diagnosis'] = [target_names[i] for i in y_train_stored]

print(f"Total training instances memorized: {X_train_stored.shape[0]}")
print(f"Target distribution: Malignant={sum(y_train_stored == 0)} | Benign={sum(y_train_stored == 1)}\n")

# 4. Filter terminal output to just 5 columns to prevent line-wrapping noise
preview_columns = list(feature_names[:5]) + ['diagnosis']

print("Terminal Preview (First 10 Features Only):")
print(df[preview_columns].head(50).round(4))

# 5. Export the entire 30-feature dataset to a CSV
csv_filename = "full_trained_data.csv"
df.to_csv(csv_filename, index=False)
print(f"\nSaved all 30 features for all {len(df)} records to '{csv_filename}'.")
print("Click on 'full_trained_data.csv' in VS Code's sidebar to view the full table cleanly.")