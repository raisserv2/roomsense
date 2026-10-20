"""RoomSense capture service: saves labelled snapshots from the IP camera."""
import csv
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# --- Settings (from the .env file) ---
CAM_URL = os.environ["CAM_URL"]
CAM_AUTH = (os.environ["CAM_USER"], os.environ["CAM_PASS"])
INTERVAL = float(os.environ.get("INTERVAL_SECONDS", "5"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/raw"))
TZ = ZoneInfo(os.environ.get("TIMEZONE", "Asia/Kolkata"))

LABELS = ["lying_in_bed", "sitting_on_bed", "at_desk", "in_room", "empty", "lights_off"]
LIGHTING = ["tube_on", "tube_off"]
MANIFEST = DATA_DIR / "frames.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capture")

# Shared state between the web app and the capture thread
state = {"session": None, "frames": 0, "last_frame": None, "last_error": None}
lock = threading.Lock()


class SessionRequest(BaseModel):
    label: str
    lighting: str


def append_manifest(row: list) -> None:
    """Add one row to frames.csv, writing the header if the file is new."""
    is_new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["path", "session_id", "label", "lighting", "timestamp"])
        writer.writerow(row)


def capture_loop() -> None:
    """Runs forever: while a session is active, save one frame every INTERVAL seconds."""
    while True:
        with lock:
            session = state["session"]
        if session:
            try:
                resp = requests.get(CAM_URL, auth=CAM_AUTH, timeout=5)
                resp.raise_for_status()
                now = datetime.now(TZ)
                folder = DATA_DIR / session["id"]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{now:%Y%m%d-%H%M%S}.jpg"
                path.write_bytes(resp.content)
                append_manifest([str(path.relative_to(DATA_DIR)), session["id"],
                                 session["label"], session["lighting"], now.isoformat()])
                with lock:
                    state["frames"] += 1
                    state["last_frame"] = resp.content
                    state["last_error"] = None
            except requests.RequestException as e:
                log.warning("Camera fetch failed: %s", e)
                with lock:
                    state["last_error"] = str(e)
            except Exception:
                log.exception("Unexpected error while saving a frame")
        time.sleep(INTERVAL)


app = FastAPI(title="RoomSense capture")
threading.Thread(target=capture_loop, daemon=True).start()


@app.post("/session/start")
def start_session(req: SessionRequest):
    if req.label not in LABELS or req.lighting not in LIGHTING:
        raise HTTPException(status_code=400, detail="Unknown label or lighting")
    session_id = datetime.now(TZ).strftime("s%Y%m%d-%H%M%S")
    with lock:
        state["session"] = {"id": session_id, "label": req.label, "lighting": req.lighting}
        state["frames"] = 0
    log.info("Session %s started: %s / %s", session_id, req.label, req.lighting)
    return {"session_id": session_id}


@app.post("/session/stop")
def stop_session():
    with lock:
        session, frames = state["session"], state["frames"]
        state["session"] = None
    log.info("Session stopped: %s (%d frames)", session["id"] if session else None, frames)
    return {"stopped": session, "frames": frames}


@app.get("/status")
def status():
    with lock:
        return {"session": state["session"], "frames": state["frames"],
                "last_error": state["last_error"], "interval_seconds": INTERVAL}


@app.get("/latest.jpg")
def latest_frame():
    with lock:
        img = state["last_frame"]
    if img is None:
        raise HTTPException(status_code=404, detail="No frame captured yet")
    return Response(content=img, media_type="image/jpeg")


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomSense capture</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px;font-size:20px}
h1{font-size:26px;margin:0 0 12px}
select,button{font-size:22px;width:100%;padding:16px;margin:6px 0;border-radius:12px;border:0}
.lab{background:#2d6cdf;color:#fff}.stop{background:#c0392b;color:#fff}
#st{padding:14px;border-radius:12px;background:#222;margin:10px 0}
img{width:100%;border-radius:12px;margin-top:10px}
</style></head><body>
<h1>RoomSense capture</h1>
<div id="st">Loading...</div>
<label>Lighting</label>
<select id="light"><option>tube_on</option><option>tube_off</option></select>
BUTTONS
<button class="stop" onclick="stopS()">&#9632; Stop</button>
<img id="img" alt="">
<script>
async function startS(label){
  const lighting=document.getElementById('light').value;
  await fetch('session/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({label:label,lighting:lighting})});
  refresh();
}
async function stopS(){await fetch('session/stop',{method:'POST'});refresh();}
async function refresh(){
  const s=await (await fetch('status')).json();
  const st=document.getElementById('st');
  if(s.session){
    st.textContent='\u25CF Recording '+s.session.label+' ('+s.session.lighting+'): '+s.frames+' frames';
    st.style.background='#1e5631';
    document.getElementById('img').src='latest.jpg?'+Date.now();
  }else{st.textContent='Not recording';st.style.background='#222';}
  if(s.last_error){st.textContent+=' | camera error: '+s.last_error;}
}
setInterval(refresh,3000);refresh();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page():
    buttons = "".join(f'<button class="lab" onclick="startS(\'{l}\')">{l}</button>' for l in LABELS)
    return PAGE.replace("BUTTONS", buttons)
