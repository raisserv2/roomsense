"""All configuration comes from environment variables (deploy/.env)."""
import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


MODEL_URI = os.environ.get("MODEL_URI", "models:/roomsense@champion")
MODEL_POLL_SECONDS = float(os.environ.get("MODEL_POLL_SECONDS", "60"))
ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))

CAMERA_ENABLED = _bool("CAMERA_ENABLED", True)
CAM_URL = os.environ.get("CAM_URL", "")
CAM_USER = os.environ.get("CAM_USER", "")
CAM_PASS = os.environ.get("CAM_PASS", "")
CAMERA_INTERVAL = float(os.environ.get("CAMERA_INTERVAL_SECONDS", "5"))

# Ground truth: while a capture session is recording, its label is the truth.
CAPTURE_STATUS_URL = os.environ.get("CAPTURE_STATUS_URL", "http://roomsense-capture:8100/status")

DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "120"))          # frames (~10 min at 5 s)
DRIFT_MIN_SAMPLES = int(os.environ.get("DRIFT_MIN_SAMPLES", "30"))
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "0.5"))     # alert when > half the window is unseen
OOD_MIN_SHARE = float(os.environ.get("OOD_MIN_SHARE", "0.005"))  # training share below which a bin is "unseen"
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE", "0.6"))

STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))
RETRAIN_TOKEN = os.environ.get("RETRAIN_TOKEN", "")

# Path prefix the frontend proxies the API under (so /api/docs works)
ROOT_PATH = os.environ.get("ROOT_PATH", "")
