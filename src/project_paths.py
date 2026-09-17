from pathlib import Path

# Project root is two levels up from this file: src/ -> project_root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure all target output directories exist
for directory in [MODELS_DIR, DATA_PROCESSED_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_path(filename: str) -> Path:
    return DATA_PROCESSED_DIR / filename


def get_figure_path(filename: str) -> Path:
    return FIGURES_DIR / filename