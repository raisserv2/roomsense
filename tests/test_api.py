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


def test_predict_rejects_empty_and_unknown_truth(client):
    assert client.post("/predict", files={"file": ("f.jpg", b"", "image/jpeg")}).status_code == 400
    r = client.post("/predict?monitor_frame=true&truth=flying", files={"file": ("f.jpg", jpeg(120), "image/jpeg")})
    assert r.status_code == 422 and "Unknown truth label" in r.json()["detail"]


def test_predict_without_a_file_is_a_validation_error(client):
    assert client.post("/predict").status_code == 422


def test_webhook_rejects_bodies_that_are_not_json(client):
    r = client.post("/hooks/alertmanager", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert client.post("/retrain/ack", content=b"[1, 2]",
                       headers={"Content-Type": "application/json"}).status_code == 400


def test_manual_retrain_request_is_idempotent(client):
    assert client.post("/retrain/request?reason=test").json() == {"raised": True}
    assert client.post("/retrain/request?reason=test").json() == {"raised": False}
    assert client.post("/retrain/ack", json={"version": "3", "promoted": False}).json()["requested"] is False


def test_health_reports_uptime_and_camera_state(client):
    h = client.get("/health").json()
    assert h["camera_enabled"] is False and h["uptime_seconds"] >= 0


def test_unknown_route_is_a_clean_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/health", "/model", "/predict", "/latest", "/latest.jpg", "/drift", "/reload", "/hooks/alertmanager",
              "/retrain", "/retrain/request", "/retrain/ack", "/metrics", "/pipeline", "/pipeline/runs",
              "/pipeline/registry", "/pipeline/rollback", "/events"]:
        assert p in paths, p
