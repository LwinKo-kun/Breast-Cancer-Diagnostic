# Breast Cancer Diagnostic Classifier

A compact machine learning project that trains a K-Nearest Neighbors (KNN) classifier on the scikit-learn breast cancer dataset and exposes it through a FastAPI inference service.

This repository is intended for learning, experimentation, and lightweight model inspection. It includes training, API serving, validation, and synthetic-data checks.

## Project goal

The project demonstrates:

- training a tabular ML model on the breast cancer dataset
- tuning KNN hyperparameters with grid search
- serializing the trained pipeline to a model artifact
- serving predictions via a FastAPI API
- checking model/data overlap and verifying behavior on unseen data

## Tech stack

- Python 3
- scikit-learn
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Joblib
- Requests

## Repository contents

- `train.py` — trains the model and saves `model.joblib`
- `main.py` — FastAPI app and prediction endpoint
- `predict.py` — simple CLI client for testing the API
- `generate_data.py` — creates synthetic novel samples
- `check_data.py` — checks for duplicates against training data
- `check_overlap.py` — validates generated rows against training data
- `inspect_data.py` — inspects the memorized training examples
- `test_api.py` — performs a live API accuracy benchmark
- `project_paths.py` — centralizes project-root paths to avoid running errors from different directories
- `model.joblib` — trained model artifact

## Setup

Create and activate the project virtual environment:

```bash
cd /home/lwin-ko/Repo/breast_cancer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train.py 30
```

This trains on 30 examples, evaluates on a held-out set, tunes KNN parameters, and saves:

```bash
model.joblib
```

The script prints the selected hyperparameters and evaluation metrics such as macro F1 and ROC-AUC.

## Run the API

Start the server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The app will automatically load the model artifact on startup.

## API endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

Response example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "features_expected": 30
}
```

### Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]}'
```

Response includes:

- prediction
- prediction_code
- confidence_pct
- class_probabilities
- nearest_neighbors_distances

## Test from the CLI

Run the interactive client:

```bash
python predict.py
```

It offers two choices:

1. random patient from the dataset
2. custom feature vector

## Validate the model and data

You can run the helper scripts:

```bash
python check_data.py
python check_overlap.py
python inspect_data.py
python test_api.py
```

These scripts help confirm:

- no immediate overlap between training and holdout samples
- generated synthetic data is sufficiently distinct
- the API performs reasonably on unseen examples

## Notes and caveats

This project is intentionally educational and experimental.

Important points:

- The model is based on KNN, which memorizes training samples.
- The project uses a deliberately small training subset for experimentation.
- The API expects exactly 30 feature values in the same order as the dataset.
- The app depends on the model artifact existing before API startup.
- The project uses project-root-aware file handling so it works reliably regardless of where it is launched.

## Typical workflow

```bash
cd /home/lwin-ko/Repo/breast_cancer
source .venv/bin/activate
python train.py 30
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## Troubleshooting

If the app fails to start:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py 30
```

If the model is missing:

```bash
python train.py 30
```

If you run scripts from outside the project directory, the project-root path fix should prevent common file-not-found issues.

## License

This project is intended for educational and experimental use.
