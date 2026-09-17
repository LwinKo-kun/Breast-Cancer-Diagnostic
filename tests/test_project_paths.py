from pathlib import Path
import unittest
from src.project_paths import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    get_project_root,
)


class TestProjectPaths(unittest.TestCase):

    def test_root_resolution(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(get_project_root(), root)

    def test_directory_locations(self):
        root = get_project_root()
        self.assertEqual(MODELS_DIR, root / "models")
        self.assertEqual(DATA_PROCESSED_DIR, root / "data" / "processed")
        self.assertEqual(FIGURES_DIR, root / "reports" / "figures")
        self.assertEqual(MODEL_PATH, root / "models" / "model.joblib")


if __name__ == "__main__":
    unittest.main()