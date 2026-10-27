"""API tests against the real FastAPI app with a tiny ONNX model."""
from tests.conftest import CLASSES, jpeg


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] and r["model_version"] == "local"


def test_model_info(client):
    m = client.get("/model").json()
    assert m["classes"] == CLASSES and m["metrics"]["test_macro_f1"] == 0.5


def test_predict(client):
    r = client.post("/predict", files={"file": ("f.jpg", jpeg(120), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] in CLASSES
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-4
    assert 0 <= j["confidence"] <= 1 and j["latency_ms"] > 0
    assert abs(j["brightness"] - 120) < 2


def test_predict_rejects_non_images(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_monitored_uploads_feed_the_drift_window(client):
    before = client.get("/drift").json()["window"]
    for _ in range(6):
        client.post("/predict?monitor_frame=true&truth=at_desk",
                    files={"file": ("f.jpg", jpeg(20), "image/jpeg")})  # far darker than training
    d = client.get("/drift").json()
    assert d["window"] == before + 6
    assert d["ood_share"] > 0.5 and d["drifting"] is True
    assert d["labelled"] >= 6 and d["live_accuracy"] is not None


def test_alert_webhook_requests_retrain_and_ack_clears_it(client):
    assert client.get("/retrain").json()["requested"] is False
    ignored = {"alerts": [{"status": "firing", "labels": {"alertname": "RoomSenseAPIDown"}}]}
    assert client.post("/hooks/alertmanager", json=ignored).json()["retrain_requested"] is False
    drift_alert = {"alerts": [{"status": "firing",
                               "labels": {"alertname": "RoomSenseDrift", "action": "retrain"},
                               "annotations": {"summary": "lights off"}}]}
    assert client.post("/hooks/alertmanager", json=drift_alert).json()["retrain_requested"] is True
    s = client.get("/retrain").json()
    assert s["requested"] and "RoomSenseDrift" in s["reason"]
    # a second firing alert does not stack a second request
    assert client.post("/hooks/alertmanager", json=drift_alert).json()["retrain_requested"] is False
    s = client.post("/retrain/ack", json={"version": "2", "promoted": True}).json()
    assert s["requested"] is False and s["history"][-1]["version"] == "2"


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    for name in ["roomsense_inference_latency_seconds", "roomsense_predictions_total",
                 "roomsense_drift_ood_share", "roomsense_drift_psi", "roomsense_model_loaded",
                 "roomsense_retrain_requests_total", "roomsense_live_accuracy"]:
        assert name in text


def test_latest_without_camera(client):
    assert client.get("/latest").status_code == 404
    assert client.get("/latest.jpg").status_code == 404


def test_reload_local_model(client):
    r = client.post("/reload?force=true").json()
    assert r["version"] == "local" and r["error"] is None
