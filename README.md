# 🎗️ Breast Cancer Diagnostic

A machine learning project that uses the **K-Nearest Neighbors (KNN)** algorithm to classify breast cancer tumors as benign or malignant.

The project includes model training, data generation, visualization, prediction, a FastAPI application, and automated tests.

> **Status:** Educational / Experimental  
> **Important:** This project is not a medical diagnostic tool and must not be used for real medical decisions.

---

## 📌 Overview

This project demonstrates how machine learning can be used to classify breast cancer data.

The system provides:

- KNN model training
- Breast cancer dataset processing
- Synthetic data generation
- Data visualization
- Model-based predictions
- FastAPI application
- Prediction history storage
- Automated testing

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scikit-learn | Machine learning |
| FastAPI | API development |
| Uvicorn | API server |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Joblib | Model saving and loading |
| Matplotlib | Data visualization |
| Pytest | Automated testing |

---

## 📂 Project Structure

```text
Breast-Cancer-Diagnostic/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       ├── prediction_history.csv
│       └── unlabeled_synthetic_data.csv
│
├── models/
│   └── model.joblib
│
├── reports/
│   └── figures/
│       ├── feature_distributions.png
│       └── knn_pca_decision_space.png
│
├── src/
│   ├── __init__.py
│   ├── project_paths.py
│   ├── train.py
│   ├── predict.py
│   ├── generate_data.py
│   └── visualize_data.py
│
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_data_integrity.py
    └── test_project_paths.py
```

### Main Files

| File | Description |
|---|---|
| `app.py` | Application entry point |
| `main.py` | Main application/API implementation |
| `requirements.txt` | Python dependencies |
| `src/train.py` | Trains the KNN model |
| `src/predict.py` | Makes predictions using the trained model |
| `src/generate_data.py` | Generates synthetic data |
| `src/visualize_data.py` | Creates data visualizations |
| `src/project_paths.py` | Defines project file paths |
| `models/model.joblib` | Saved machine learning model |

---

## 🧠 Machine Learning Model

The project uses the **K-Nearest Neighbors (KNN)** classification algorithm.

KNN predicts the class of a new sample by comparing it with nearby samples in the training dataset.

### Classification Classes

The model classifies tumors into:

- **Benign**
- **Malignant**

### Input Features

The Breast Cancer Wisconsin Diagnostic dataset contains **30 numerical features** describing characteristics of cell nuclei.

These features are used by the model to make predictions.

---

## 📊 Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through `scikit-learn`.

The dataset contains numerical measurements extracted from breast tissue samples.

The data is used for:

- Model training
- Model evaluation
- Prediction experiments
- Visualization
- Synthetic data generation

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LwinKo-kun/Breast-Cancer-Diagnostic.git
cd Breast-Cancer-Diagnostic
```

### 2. Create a Virtual Environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run the training script:

```bash
python -m src.train
```

The training process prepares the dataset and trains the KNN classification model.

The trained model is stored in:

```text
models/model.joblib
```

> Check `src/train.py` for the available training options and parameters.

---

## 🧪 Generate Synthetic Data

The project includes a script for generating synthetic data.

Run:

```bash
python -m src.generate_data
```

Generated data is stored in:

```text
data/processed/unlabeled_synthetic_data.csv
```

Synthetic data is intended for experimentation and testing. It should not be treated as real patient data.

---

## 📈 Visualize the Dataset

Generate visualizations using:

```bash
python -m src.visualize_data
```

The generated figures are stored in:

```text
reports/figures/
```

### Available Visualizations

#### Feature Distributions

```text
reports/figures/feature_distributions.png
```

This figure shows the distribution of selected dataset features.

#### KNN PCA Decision Space

```text
reports/figures/knn_pca_decision_space.png
```

This figure visualizes the KNN classification space after applying Principal Component Analysis (PCA).

---

## 🚀 Run the Application

The project includes a Python application and API implementation.

Start the application according to the entry point defined in `app.py` and `main.py`.

For a FastAPI application, the server can typically be started using:

```bash
uvicorn main:app --reload
```

The API is then available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation, if enabled, is available at:

```text
http://127.0.0.1:8000/docs
```

> The exact startup command depends on how the application is configured. Check `app.py` and `main.py` for the available entry points.

---

## 🔍 Prediction

The project includes a prediction script:

```bash
python -m src.predict
```

The prediction module uses the trained model to classify input data.

The saved model is loaded from:

```text
models/model.joblib
```

Prediction results may be recorded in:

```text
data/processed/prediction_history.csv
```

---

## 🧪 Testing

The project uses **Pytest** for automated testing.

Run all tests:

```bash
pytest
```

### Test Files

| File | Purpose |
|---|---|
| `tests/test_api.py` | Tests API functionality |
| `tests/test_data_integrity.py` | Checks data integrity |
| `tests/test_project_paths.py` | Tests project path configuration |

You can also run a specific test file:

```bash
pytest tests/test_api.py
```

---

## 🔄 Project Workflow

```text
Breast Cancer Dataset
        │
        ▼
Data Preparation
        │
        ▼
KNN Model Training
        │
        ▼
Model Evaluation
        │
        ▼
model.joblib
        │
        ▼
Prediction Module
        │
        ▼
FastAPI Application
        │
        ▼
Prediction Results
        │
        ▼
Prediction History
```

---

## 📁 Generated Files

| File | Description |
|---|---|
| `models/model.joblib` | Trained machine learning model |
| `data/processed/unlabeled_synthetic_data.csv` | Generated synthetic data |
| `data/processed/prediction_history.csv` | Prediction history |
| `reports/figures/feature_distributions.png` | Feature distribution visualization |
| `reports/figures/knn_pca_decision_space.png` | PCA decision-space visualization |

---

## ⚠️ Limitations

- This project is intended for learning and experimentation.
- The model is not clinically validated.
- Predictions are not medical diagnoses.
- Synthetic data does not represent real patient records.
- Model performance depends on the dataset and training process.
- The application should not be used to make medical decisions.

---

## 🎯 Learning Objectives

This project helps demonstrate:

- Machine learning classification
- KNN algorithms
- Dataset processing
- Synthetic data generation
- Data visualization
- Model serialization
- FastAPI development
- Automated testing
- Python project organization

---

## 👨‍💻 Author

**Lwin Ko**

GitHub:  
https://github.com/LwinKo-kun

Repository:  
https://github.com/LwinKo-kun/Breast-Cancer-Diagnostic

---

## 📄 License

No license has been specified in the repository.

If you plan to distribute or reuse this project, consider adding an appropriate open-source license.