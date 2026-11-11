"""Capture service: label rules, dataset statistics, /stats and /metrics."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """The real capture app, loaded under its own module name (the API package is also called `app`)."""
    data = tmp_path_factory.mktemp("capture-data")
    mp = pytest.MonkeyPatch()
    for key, value in {"CAM_URL": "http://127.0.0.1:9/shot.jpg", "CAM_USER": "u", "CAM_PASS": "p",
                       "DATA_DIR": str(data / "raw"), "TRASH_DIR": str(data / "trash"),
                       "INTERVAL_SECONDS": "3600"}.items():
        mp.setenv(key, value)
    mp.syspath_prepend(str(ROOT / "capture"))
    spec = importlib.util.spec_from_file_location("capture_app", ROOT / "capture" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from fastapi.testclient import TestClient
    yield TestClient(module.app), sys.modules["storage"]
    mp.undo()


def test_impossible_and_unknown_labels_are_refused(capture):
    client, storage = capture
    storage.validate_combo("at_desk", "tube_on")
    with pytest.raises(ValueError, match="impossible"):
        storage.validate_combo("lights_off", "tube_on")
    r = client.post("/session/start", json={"label": "dancing", "lighting": "tube_on"})
    assert r.status_code == 400 and "Unknown label" in r.json()["detail"]


def test_stats_count_frames_and_sessions_per_class(capture):
    client, storage = capture
    for i in range(3):
        storage.append_frame({"path": f"s20261101-100000/{i}.jpg", "session_id": "s20261101-100000",
                              "label": "at_desk", "lighting": "tube_on", "timestamp": f"2026-11-01T10:00:0{i}+05:30"})
    storage.append_frame({"path": "s20261101-230000/0.jpg", "session_id": "s20261101-230000",
                          "label": "lights_off", "lighting": "tube_off", "timestamp": "2026-11-01T23:00:00+05:30"})
    s = client.get("/stats").json()
    assert s["frames_total"] == 4 and s["sessions_total"] == 2
    assert s["per_class"]["at_desk"] == {"frames": 3, "sessions": 1}
    assert s["per_class"]["empty"] == {"frames": 0, "sessions": 0}
    assert s["last_frame_at"].startswith("2026-11-01T23") and s["recording"] is None


def test_session_start_stop_and_metrics(capture):
    client, _ = capture
    assert client.post("/session/start", json={"label": "empty", "lighting": "tube_on"}).status_code == 200
    assert client.get("/status").json()["session"]["label"] == "empty"
    text = client.get("/metrics").text
    assert "roomsense_capture_session_active 1.0" in text
    assert 'roomsense_capture_dataset_frames{label="at_desk"} 3.0' in text
    assert "roomsense_capture_camera_errors_total" in text
    client.post("/session/stop")
    assert "roomsense_capture_session_active 0.0" in client.get("/metrics").text
