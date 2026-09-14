import unittest
from pathlib import Path

from project_paths import MODEL_PATH, get_project_root


class ProjectPathTests(unittest.TestCase):
    def test_model_path_is_stored_in_project_root(self):
        root = Path(__file__).resolve().parent
        self.assertEqual(get_project_root(), root)
        self.assertEqual(MODEL_PATH.parent, root)
        self.assertEqual(MODEL_PATH.name, "model.joblib")


if __name__ == "__main__":
    unittest.main()
