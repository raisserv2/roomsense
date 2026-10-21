"""Review tools: browse sessions, spot mistakes, trash or relabel frames."""
import io
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

import storage

log = logging.getLogger("review")
router = APIRouter()

SESSION_RE = re.compile(r"^s\d{8}-\d{6}$")
FRAME_RE = re.compile(r"^s\d{8}-\d{6}/\d{8}-\d{6}\.jpg$")
REVIEW_PAGE = (Path(__file__).parent / "review.html").read_text()
_activity_cache = {}


class TrashRequest(BaseModel):
    paths: list[str]
    reason: str = "manual"


class RelabelRequest(BaseModel):
    paths: list[str]
    label: str


def _frame_path(sid: str, name: str) -> Path:
    rel = f"{sid}/{name}"
    path = storage.DATA_DIR / rel
    if not FRAME_RE.match(rel) or not path.exists():
        raise HTTPException(status_code=404, detail="No such frame")
    return path


def _session_frames(sid: str) -> list:
    if not SESSION_RE.match(sid):
        raise HTTPException(status_code=404, detail="No such session")
    frames = [r for r in storage.read_manifest() if r["session_id"] == sid]
    return sorted(frames, key=lambda r: r["timestamp"])


def _signature(path: Path):
    """A tiny 32x24 greyscale copy of a frame, used to measure change and brightness."""
    try:
        with Image.open(path) as im:
            im.draft("L", (80, 60))
            return im.convert("L").resize((32, 24)).tobytes()
    except Exception:
        log.warning("Could not read %s", path)
        return None


@router.get("/review", response_class=HTMLResponse)
def review_page():
    return REVIEW_PAGE


@router.get("/api/meta")
def meta():
    return {"labels": storage.LABELS, "lighting": storage.LIGHTING}


@router.get("/api/sessions")
def list_sessions():
    groups = {}
    for r in storage.read_manifest():
        g = groups.setdefault(r["session_id"], {
            "session_id": r["session_id"], "frames": 0, "labels": set(), "lighting": set(),
            "start": r["timestamp"], "end": r["timestamp"]})
        g["frames"] += 1
        g["labels"].add(r["label"])
        g["lighting"].add(r["lighting"])
        g["start"], g["end"] = min(g["start"], r["timestamp"]), max(g["end"], r["timestamp"])
    out = [{**g, "labels": sorted(g["labels"]), "lighting": sorted(g["lighting"])} for g in groups.values()]
    return sorted(out, key=lambda g: g["start"], reverse=True)


@router.get("/api/sessions/{sid}/frames")
def session_frames(sid: str):
    return _session_frames(sid)


@router.get("/api/sessions/{sid}/activity")
def session_activity(sid: str):
    """Per frame: how much it differs from the previous frame, and its brightness (0-1)."""
    frames = _session_frames(sid)
    key = (sid, len(frames), frames[-1]["path"] if frames else "")
    if key in _activity_cache:
        return _activity_cache[key]
    sigs = [_signature(storage.DATA_DIR / r["path"]) for r in frames]
    diff, brightness = [], []
    for i, sig in enumerate(sigs):
        brightness.append(round(sum(sig) / len(sig) / 255, 3) if sig else 0)
        prev = sigs[i - 1] if i else None
        diff.append(round(sum(abs(a - b) for a, b in zip(sig, prev)) / len(sig), 2) if sig and prev else 0)
    result = {"diff": diff, "brightness": brightness}
    _activity_cache[key] = result
    return result


@router.get("/frame/{sid}/{name}")
def frame(sid: str, name: str):
    return FileResponse(_frame_path(sid, name), media_type="image/jpeg")


@router.get("/thumb/{sid}/{name}")
def thumb(sid: str, name: str):
    path = _frame_path(sid, name)
    with Image.open(path) as im:
        im.draft("RGB", (320, 240))
        small = im.convert("RGB")
        small.thumbnail((240, 180))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=70)
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@router.post("/api/trash")
def trash(req: TrashRequest):
    try:
        return storage.trash_frames(req.paths, req.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/api/trash")
def list_trash():
    return storage.trash_actions()


@router.post("/api/trash/{action_id}/restore")
def restore(action_id: str):
    try:
        return {"frames": storage.restore(action_id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/relabel")
def relabel(req: RelabelRequest):
    try:
        return {"frames": storage.relabel(req.paths, req.label)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))


class LightingRequest(BaseModel):
    paths: list[str]
    lighting: str


@router.post("/api/lighting")
def change_lighting(req: LightingRequest):
    try:
        return {"frames": storage.set_lighting(req.paths, req.lighting)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))
