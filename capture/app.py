"""RoomSense capture service: saves labelled snapshots from the IP camera."""
import logging
import os
import threading
import time

import capmetrics as M
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

import storage
from review import router as review_router

# --- Settings (from the .env file) ---
CAM_URL = os.environ["CAM_URL"]
CAM_AUTH = (os.environ["CAM_USER"], os.environ["CAM_PASS"])
INTERVAL = float(os.environ.get("INTERVAL_SECONDS", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("capture")

# Shared state between the web app and the capture thread
state = {"session": None, "frames": 0, "last_frame": None, "last_error": None}
state_lock = threading.Lock()


class SessionRequest(BaseModel):
    label: str
    lighting: str


def capture_loop() -> None:
    """Runs forever: while a session is active, save one frame every INTERVAL seconds."""
    while True:
        with state_lock:
            session = state["session"]
        if session:
            try:
                t0 = time.perf_counter()
                resp = requests.get(CAM_URL, auth=CAM_AUTH, timeout=5)
                resp.raise_for_status()
                M.FETCH_SECONDS.observe(time.perf_counter() - t0)
                now = storage.now()
                folder = storage.DATA_DIR / session["id"]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{now:%Y%m%d-%H%M%S}.jpg"
                path.write_bytes(resp.content)
                storage.append_frame({
                    "path": str(path.relative_to(storage.DATA_DIR)), "session_id": session["id"],
                    "label": session["label"], "lighting": session["lighting"],
                    "timestamp": now.isoformat()})
                with state_lock:
                    state["frames"] += 1
                    state["last_frame"] = resp.content
                    state["last_error"] = None
                M.FRAMES.labels(session["label"], session["lighting"]).inc()
                M.LAST_FRAME.set(time.time())
            except requests.RequestException as e:
                log.warning("Camera fetch failed: %s", e)
                M.CAMERA_ERRORS.inc()
                with state_lock:
                    state["last_error"] = str(e)
            except Exception as e:  # noqa: BLE001  (the loop must survive a full disk or a bad CSV)
                log.exception("Unexpected error while saving a frame")
                M.SAVE_ERRORS.inc()
                with state_lock:
                    state["last_error"] = f"could not save the frame: {e}"
        time.sleep(INTERVAL)


app = FastAPI(title="RoomSense capture")
app.include_router(review_router)
threading.Thread(target=capture_loop, daemon=True).start()


@app.post("/session/start")
def start_session(req: SessionRequest):
    try:
        storage.validate_combo(req.label, req.lighting)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    session_id = storage.now().strftime("s%Y%m%d-%H%M%S")
    with state_lock:
        state["session"] = {"id": session_id, "label": req.label, "lighting": req.lighting}
        state["frames"] = 0
    storage.active["session_id"] = session_id
    M.SESSION_ACTIVE.set(1)
    M.SESSIONS.labels(req.label).inc()
    log.info("Session %s started: %s / %s", session_id, req.label, req.lighting)
    return {"session_id": session_id}


@app.post("/session/stop")
def stop_session():
    with state_lock:
        session, frames = state["session"], state["frames"]
        state["session"] = None
    storage.active["session_id"] = None
    M.SESSION_ACTIVE.set(0)
    log.info("Session stopped: %s (%d frames)", session["id"] if session else None, frames)
    return {"stopped": session, "frames": frames}


@app.get("/status")
def status():
    with state_lock:
        return {"session": state["session"], "frames": state["frames"],
                "last_error": state["last_error"], "interval_seconds": INTERVAL}


@app.get("/stats")
def stats():
    """Dataset totals for the pipeline console: frames and sessions per class, last frame, free disk."""
    with state_lock:
        recording = state["session"]
    return {**storage.dataset_stats(), "recording": recording, "interval_seconds": INTERVAL}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint. Dataset gauges are refreshed from the (cached) stats."""
    try:
        s = storage.dataset_stats()
        for label, c in s["per_class"].items():
            M.DATASET_FRAMES.labels(label).set(c["frames"])
            M.DATASET_SESSIONS.labels(label).set(c["sessions"])
        if s["disk_free_bytes"] is not None:
            M.DISK_FREE.set(s["disk_free_bytes"])
    except OSError:
        log.warning("Could not refresh the dataset gauges")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/latest.jpg")
def latest_frame():
    with state_lock:
        img = state["last_frame"]
    if img is None:
        raise HTTPException(status_code=404, detail="No frame captured yet")
    return Response(content=img, media_type="image/jpeg")


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomSense capture</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px;font-size:20px}
h1{font-size:26px;margin:0 0 12px} a{color:#7fb0ff}
select,button{font-size:22px;width:100%;padding:16px;margin:6px 0;border-radius:12px;border:0}
.lab{background:#2d6cdf;color:#fff}.stop{background:#c0392b;color:#fff}
#st{padding:14px;border-radius:12px;background:#222;margin:10px 0}
img{width:100%;border-radius:12px;margin-top:10px}
</style></head><body>
<h1>RoomSense capture</h1>
<div id="st">Loading...</div>
<label for="light">Lighting</label>
<select id="light">LIGHTOPTIONS</select>
BUTTONS
<button class="stop" onclick="stopS()">&#9632; Stop</button>
<p><a href="review">Review and fix frames &rarr;</a></p>
<img id="img" alt="">
<script>
async function startS(label){
  const lighting=document.getElementById('light').value;
  const r=await fetch('session/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({label:label,lighting:lighting})});
  if(!r.ok){const e=await r.json();document.getElementById('st').textContent='Not started: '+e.detail;return;}
  refresh();
}
async function stopS(){await fetch('session/stop',{method:'POST'});refresh();}
async function refresh(){
  const s=await (await fetch('status')).json();
  const st=document.getElementById('st');
  if(s.session){
    st.textContent='\\u25CF Recording '+s.session.label+' ('+s.session.lighting+'): '+s.frames+' frames';
    st.style.background='#1e5631';
    document.getElementById('img').src='latest.jpg?'+Date.now();
  }else if(!st.textContent.startsWith('Not started')){st.textContent='Not recording';st.style.background='#222';}
  if(s.last_error){st.textContent+=' | camera error: '+s.last_error;}
}
setInterval(refresh,3000);refresh();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page():
    buttons = "".join(f'<button class="lab" onclick="startS(\'{lab}\')">{lab}</button>' for lab in storage.LABELS)
    lights = "".join(f"<option>{opt}</option>" for opt in storage.LIGHTING)
    return PAGE.replace("BUTTONS", buttons).replace("LIGHTOPTIONS", lights)
