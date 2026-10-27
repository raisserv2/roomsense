"""RoomSense inference API (runs on nova).

GET  /health              liveness + whether a model is loaded
GET  /model               details of the serving model (registry version, metrics)
POST /predict             classify an uploaded image (?monitor=true feeds the drift window)
GET  /latest              last camera prediction (+ ground truth while a capture session runs)
GET  /latest.jpg          last camera frame
GET  /drift               drift window summary (PSI, brightness, confidence, live accuracy)
POST /reload              re-check the registry now (?force=true reloads even if unchanged)
POST /hooks/alertmanager  Alertmanager webhook: a firing drift alert requests a retrain
GET  /retrain             retrain request status (polled by scripts/retrain.py on the Legion)
POST /retrain/ack         the Legion reports a finished retrain
GET  /metrics             Prometheus metrics
"""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from roomsense import frames

from . import settings as S
from .model import ModelManager
from .monitor import Monitor
from .state import RetrainState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

models = ModelManager(S.MODEL_URI)
monitor = Monitor(models)
retrain = RetrainState(S.STATE_DIR / "retrain.json")
_stop = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load()
    if S.MODEL_POLL_SECONDS > 0:
        threading.Thread(target=models.poll_forever, args=(S.MODEL_POLL_SECONDS, _stop), daemon=True).start()
    if S.CAMERA_ENABLED and S.CAM_URL:
        threading.Thread(target=monitor.camera_forever, args=(_stop,), daemon=True).start()
    yield
    _stop.set()


app = FastAPI(title="RoomSense API", lifespan=lifespan, root_path=S.ROOT_PATH)


@app.get("/health")
def health():
    m = models.current()
    return {"status": "ok", "model_loaded": m is not None, "model_version": m.version if m else None,
            "model_error": models.error, "camera_error": monitor.camera_error}


@app.get("/model")
def model_info():
    m = models.current()
    if m is None:
        raise HTTPException(503, f"No model loaded: {models.error}")
    return m.info()


@app.post("/predict")
async def predict(file: UploadFile = File(...), monitor_frame: bool = False, truth: str = None):
    """Classify one image. `monitor_frame=true` also feeds the drift window (used by the drift demo)."""
    if models.current() is None:
        raise HTTPException(503, f"No model loaded: {models.error}")
    data = await file.read()
    try:
        img = frames.decode_jpeg(data)
    except ValueError:
        raise HTTPException(400, "Not an image")
    if monitor_frame:
        return monitor.observe(img, "upload", truth=truth)
    res = models.current().predict(img)
    res.pop("_latency_s")
    res["brightness"] = frames.brightness(img)
    res["model_version"] = models.current().version
    return res


@app.get("/latest")
def latest():
    if monitor.latest is None:
        raise HTTPException(404, monitor.camera_error or "No camera prediction yet")
    return {**monitor.latest, "camera_error": monitor.camera_error}


@app.get("/latest.jpg")
def latest_jpg():
    if monitor.latest_jpeg is None:
        raise HTTPException(404, "No camera frame yet")
    return Response(monitor.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/drift")
def drift():
    return monitor.drift_summary()


@app.post("/reload")
def reload(force: bool = False):
    changed = models.load(force=force)
    m = models.current()
    return {"reloaded": changed, "version": m.version if m else None, "error": models.error}


@app.post("/hooks/alertmanager")
async def alertmanager_hook(request: Request):
    payload = await request.json()
    raised = []
    for a in payload.get("alerts", []):
        name = (a.get("labels") or {}).get("alertname", "")
        if a.get("status") == "firing" and (a.get("labels") or {}).get("action") == "retrain":
            reason = f"{name}: {(a.get('annotations') or {}).get('summary', '')}".strip(": ")
            if retrain.request(reason):
                raised.append(reason)
                log.warning("Retrain requested: %s", reason)
    return {"retrain_requested": bool(raised), "reasons": raised}


@app.get("/retrain")
def retrain_status():
    return retrain.snapshot()


@app.post("/retrain/request")
def retrain_request(reason: str = "manual", x_token: str = Header(default="")):
    _check_token(x_token)
    return {"raised": retrain.request(reason)}


@app.post("/retrain/ack")
async def retrain_ack(request: Request, x_token: str = Header(default="")):
    _check_token(x_token)
    detail = await request.json() if await request.body() else {}
    retrain.ack(detail)
    models.load()  # pick up a newly promoted champion straight away
    return retrain.snapshot()


def _check_token(token: str) -> None:
    if S.RETRAIN_TOKEN and token != S.RETRAIN_TOKEN:
        raise HTTPException(403, "Bad X-Token")


@app.get("/metrics")
def metrics():
    monitor.update_drift_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
