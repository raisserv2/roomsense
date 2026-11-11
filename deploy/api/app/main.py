"""RoomSense inference API (runs on nova).

The full contract (inputs, outputs, error codes) is in docs/03-lld.md; the
interactive version is served at /docs.

Live room
  GET  /health              liveness, model loaded?, dependencies
  GET  /model               details of the serving model (registry version, metrics)
  POST /predict             classify an uploaded image (?monitor_frame=true feeds the drift window)
  GET  /latest              last camera prediction (+ ground truth while a capture session runs)
  GET  /latest.jpg          last camera frame
  GET  /drift               drift window summary (unseen-lighting share, PSI, confidence, live accuracy)
Pipeline console
  GET  /pipeline            DAG, latest pipeline run per stage, throughput, data ingestion
  GET  /pipeline/runs       pipeline runs (succeeded / failed) and training runs
  GET  /pipeline/registry   model versions, champion, rollback target
  POST /pipeline/rollback   move the champion alias back to an earlier version
  GET  /events              activity log (alerts, retrains, model loads, failures)
Model lifecycle
  POST /reload              re-check the registry now (?force=true reloads even if unchanged)
  POST /hooks/alertmanager  Alertmanager webhook: a firing drift alert requests a retrain
  GET  /retrain             retrain request status (polled by scripts/retrain.py on the Legion)
  POST /retrain/request     raise the retrain flag by hand
  POST /retrain/ack         the Legion reports a finished retrain
Monitoring
  GET  /metrics             Prometheus metrics
"""
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from roomsense import frames, registry

from . import metrics as M
from . import settings as S
from .model import ModelManager
from .monitor import Monitor
from .pipeline import PipelineStore
from .state import EventLog, RetrainState

logging.basicConfig(level=S.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # same limit as the frontend's nginx (client_max_body_size)
STARTED_AT = time.time()

events = EventLog(S.STATE_DIR / "events.json")
models = ModelManager(S.MODEL_URI, on_event=events.add)
monitor = Monitor(models, on_event=events.add)
retrain = RetrainState(S.STATE_DIR / "retrain.json")
pipeline = PipelineStore()
_stop = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background threads (registry poll, camera loop, pipeline poll); stop them on shutdown."""
    models.load()
    if S.MODEL_POLL_SECONDS > 0:
        threading.Thread(target=models.poll_forever, args=(S.MODEL_POLL_SECONDS, _stop), daemon=True).start()
    if S.CAMERA_ENABLED and S.CAM_URL:
        threading.Thread(target=monitor.camera_forever, args=(_stop,), daemon=True).start()
    if S.PIPELINE_POLL_SECONDS > 0:
        threading.Thread(target=pipeline.poll_forever, args=(S.PIPELINE_POLL_SECONDS, _stop), daemon=True).start()
    log.info("API started (model uri %s)", S.MODEL_URI)
    yield
    _stop.set()


app = FastAPI(title="RoomSense API", version="2.2", lifespan=lifespan, root_path=S.ROOT_PATH,
              description="Inference, drift monitoring and the pipeline console of RoomSense.")
if S.CORS_ORIGINS:  # only needed when the frontend is served from a different origin than the API
    app.add_middleware(CORSMiddleware, allow_origins=S.CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["X-Token", "Content-Type"])


@app.middleware("http")
async def record_request(request: Request, call_next):
    """Count and time every request, labelled by route template (keeps label cardinality small)."""
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        M.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        M.HTTP_LATENCY.labels(path).observe(time.perf_counter() - t0)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Last line of defence: log the traceback, answer with a clean JSON error instead of a stack trace."""
    error_id = uuid.uuid4().hex[:8]
    log.exception("unhandled error %s on %s %s", error_id, request.method, request.url.path)
    events.add("api", f"Internal error on {request.url.path} (id {error_id})", "error")
    return JSONResponse(status_code=500, content={"detail": "Internal error", "error_id": error_id})


def _check_token(token: str) -> None:
    """Protected actions need the shared secret when RETRAIN_TOKEN is set."""
    if S.RETRAIN_TOKEN and token != S.RETRAIN_TOKEN:
        raise HTTPException(403, "Bad X-Token")


# --- live room --------------------------------------------------------------
@app.get("/health", tags=["live"], summary="Liveness and readiness")
def health():
    m = models.current()
    return {"status": "ok", "model_loaded": m is not None, "model_version": m.version if m else None,
            "model_error": models.error, "camera_error": monitor.camera_error,
            "camera_enabled": bool(S.CAMERA_ENABLED and S.CAM_URL), "uptime_seconds": round(time.time() - STARTED_AT)}


@app.get("/model", tags=["live"], summary="The model that is serving")
def model_info():
    m = models.current()
    if m is None:
        raise HTTPException(503, f"No model loaded: {models.error}")
    return m.info()


@app.post("/predict", tags=["live"], summary="Classify one uploaded image")
async def predict(file: UploadFile = File(...), monitor_frame: bool = False, truth: str = None):
    """Classify one image. `monitor_frame=true` also feeds the drift window (used by the drift demo)."""
    model = models.current()
    if model is None:
        raise HTTPException(503, f"No model loaded: {models.error}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        img = frames.decode_jpeg(data)
    except ValueError:
        raise HTTPException(400, "Not an image") from None
    if truth is not None and truth not in model.classes:
        raise HTTPException(422, f"Unknown truth label '{truth}'. Known: {model.classes}")
    if monitor_frame:
        return monitor.observe(img, "upload", truth=truth)
    res = model.predict(img)
    res.pop("_latency_s")
    res["brightness"] = frames.brightness(img)
    res["model_version"] = model.version
    return res


@app.get("/latest", tags=["live"], summary="Last camera prediction")
def latest():
    if monitor.latest is None:
        raise HTTPException(404, monitor.camera_error or "No camera prediction yet")
    return {**monitor.latest, "camera_error": monitor.camera_error}


@app.get("/latest.jpg", tags=["live"], summary="Last camera frame (JPEG)")
def latest_jpg():
    if monitor.latest_jpeg is None:
        raise HTTPException(404, "No camera frame yet")
    return Response(monitor.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/drift", tags=["live"], summary="Drift window summary")
def drift():
    return monitor.drift_summary()


# --- pipeline console -------------------------------------------------------
@app.get("/pipeline", tags=["pipeline"], summary="DAG, latest run per stage, throughput, ingestion")
def pipeline_overview(refresh: bool = False):
    snap = pipeline.refresh() if refresh else pipeline.snapshot()
    return {k: snap[k] for k in ("generated_at", "mlflow_up", "errors", "dag", "stage_info", "latest",
                                 "last_success", "summary", "ingestion")} | {"retrain": retrain.snapshot()}


@app.get("/pipeline/runs", tags=["pipeline"], summary="Pipeline runs and training runs")
def pipeline_runs(limit: int = 20):
    if not 1 <= limit <= 50:
        raise HTTPException(422, "limit must be between 1 and 50")
    snap = pipeline.snapshot()
    return {"generated_at": snap["generated_at"], "summary": snap["summary"],
            "pipeline_runs": snap["runs"][:limit], "training_runs": snap["training_runs"][:limit]}


@app.get("/pipeline/registry", tags=["pipeline"], summary="Model versions and the champion")
def pipeline_registry():
    snap = pipeline.snapshot()
    versions = snap["versions"]
    champion = next((v for v in versions if v["is_champion"]), None)
    target = None
    if champion:
        prev = champion.get("previous_champion")
        older = [v["version"] for v in versions if int(v["version"]) < int(champion["version"])]
        target = prev if prev in {v["version"] for v in versions} else (older[0] if older else None)
    m = models.current()
    return {"model": S.MODEL_NAME, "alias": S.MODEL_ALIAS, "champion": champion["version"] if champion else None,
            "serving": m.version if m else None, "rollback_target": target, "versions": versions}


@app.post("/pipeline/rollback", tags=["pipeline"], summary="Roll the champion back")
def pipeline_rollback(version: str = None, reason: str = "rollback from the web UI", x_token: str = Header(default="")):
    """Move the champion alias to `version` (default: the version the champion replaced) and reload."""
    _check_token(x_token)
    client = pipeline._client_factory()
    if client is None:
        raise HTTPException(503, "MLflow is not configured")
    try:
        result = registry.rollback(client, S.MODEL_NAME, S.MODEL_ALIAS, version, reason)
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    except Exception as e:  # noqa: BLE001  (MLflow unreachable)
        log.warning("rollback failed: %s", e)
        raise HTTPException(503, f"MLflow is not answering: {str(e)[:200]}") from None
    M.ROLLBACKS.inc()
    log.warning("Rolled back: champion v%s -> v%s (%s)", result["rolled_back_from"], result["champion"], reason)
    events.add("rollback", f"Rolled back from v{result['rolled_back_from']} to v{result['champion']}", "warning")
    result["reloaded"] = models.load()
    pipeline.refresh()
    return result


@app.get("/events", tags=["pipeline"], summary="Activity log")
def activity(limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit must be between 1 and 200")
    return {"events": events.recent(limit)}


# --- model lifecycle --------------------------------------------------------
@app.post("/reload", tags=["lifecycle"], summary="Check the registry for a new champion now")
def reload(force: bool = False):
    changed = models.load(force=force)
    m = models.current()
    return {"reloaded": changed, "version": m.version if m else None, "error": models.error}


@app.post("/hooks/alertmanager", tags=["lifecycle"], summary="Alertmanager webhook")
async def alertmanager_hook(request: Request):
    """Alerts labelled `action=retrain` raise the retrain flag; everything is written to the activity log."""
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(400, "Body must be JSON") from None
    raised = []
    for a in (payload or {}).get("alerts", []):
        labels, notes = a.get("labels") or {}, a.get("annotations") or {}
        name, firing = labels.get("alertname", "alert"), a.get("status") == "firing"
        events.add("alert", f"{name} {'firing' if firing else 'resolved'}: {notes.get('summary', '')}".strip(": "),
                   "warning" if firing else "success")
        if firing and labels.get("action") == "retrain":
            reason = f"{name}: {notes.get('summary', '')}".strip(": ")
            if retrain.request(reason):
                raised.append(reason)
                log.warning("Retrain requested: %s", reason)
                events.add("retrain", f"Retrain requested: {reason}", "warning")
    return {"retrain_requested": bool(raised), "reasons": raised}


@app.get("/retrain", tags=["lifecycle"], summary="Retrain flag and history")
def retrain_status():
    return retrain.snapshot()


@app.post("/retrain/request", tags=["lifecycle"], summary="Raise the retrain flag by hand")
def retrain_request(reason: str = "manual", x_token: str = Header(default="")):
    _check_token(x_token)
    raised = retrain.request(reason)
    if raised:
        events.add("retrain", f"Retrain requested: {reason}", "warning")
    return {"raised": raised}


@app.post("/retrain/ack", tags=["lifecycle"], summary="The training machine reports a finished retrain")
async def retrain_ack(request: Request, x_token: str = Header(default="")):
    _check_token(x_token)
    try:
        detail = await request.json() if await request.body() else {}
    except ValueError:
        raise HTTPException(400, "Body must be JSON") from None
    if not isinstance(detail, dict):
        raise HTTPException(400, "Body must be a JSON object")
    retrain.ack(detail)
    events.add("retrain", f"Retrain finished: model v{detail.get('version', '?')} "
               f"{'promoted' if detail.get('promoted') else 'not promoted'}", "success")
    models.load()  # pick up a newly promoted champion straight away
    pipeline.refresh()
    return retrain.snapshot()


# --- monitoring -------------------------------------------------------------
@app.get("/metrics", tags=["monitoring"], summary="Prometheus metrics")
def metrics():
    monitor.update_drift_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
