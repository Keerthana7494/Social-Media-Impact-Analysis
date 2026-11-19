from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET = "Overall_Impact"
ID_COLUMN = "Student_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.20

for folder in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
