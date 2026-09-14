from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.joblib"


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_path(filename: str) -> Path:
    return PROJECT_ROOT / filename
