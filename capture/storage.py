"""Shared storage: the frame manifest, the trash and the curation log.

Every change to frames.csv goes through this module and its lock, so the
capture loop and the review tools never write the file at the same time.
"""
import csv
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("storage")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/raw"))
TRASH_DIR = Path(os.environ.get("TRASH_DIR", "/data/trash"))
TZ = ZoneInfo(os.environ.get("TIMEZONE", "Asia/Kolkata"))

MANIFEST = DATA_DIR / "frames.csv"
CURATION_LOG = DATA_DIR / "curation_log.csv"
TRASH_ROWS = TRASH_DIR / "trashed_rows.csv"

FIELDS = ["path", "session_id", "label", "lighting", "timestamp"]
LOG_FIELDS = ["timestamp", "action_id", "action", "sessions", "frames", "detail"]
LABELS = ["lying_in_bed", "sitting_on_bed", "at_desk", "in_room", "empty", "lights_off"]
LIGHTING = ["tube_on", "tube_off"]
INVALID_COMBOS = {("lights_off", "tube_on")}  # a dark room with the light on cannot happen

lock = threading.RLock()
active = {"session_id": None}  # set by the capture service while a session records


def now() -> datetime:
    return datetime.now(TZ)


def validate_combo(label: str, lighting: str) -> None:
    """Raise ValueError for an unknown or impossible label/lighting pair."""
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    if lighting not in LIGHTING:
        raise ValueError(f"Unknown lighting: {lighting}")
    if (label, lighting) in INVALID_COMBOS:
        raise ValueError(f"{label} with {lighting} is impossible")


def _append_csv(path: Path, fields: list, row: dict) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def append_frame(row: dict) -> None:
    with lock:
        _append_csv(MANIFEST, FIELDS, row)


def read_manifest() -> list:
    with lock:
        if not MANIFEST.exists():
            return []
        with MANIFEST.open(newline="") as f:
            return list(csv.DictReader(f))


def _write_manifest(rows: list) -> None:
    """Rewrite frames.csv safely: write a temporary file, then swap it in."""
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["timestamp"]))
    os.replace(tmp, MANIFEST)


def _log(action: str, action_id: str, sessions: set, count: int, detail: str) -> None:
    _append_csv(CURATION_LOG, LOG_FIELDS, {
        "timestamp": now().isoformat(), "action_id": action_id, "action": action,
        "sessions": " ".join(sorted(sessions)), "frames": count, "detail": detail})


def _check_not_recording(rows: list) -> None:
    sid = active["session_id"]
    if sid and any(r["session_id"] == sid for r in rows):
        raise PermissionError("That session is still recording. Stop it first.")


def trash_frames(paths: list, reason: str) -> dict:
    """Move frames into the trash folder and drop their rows from frames.csv."""
    wanted = set(paths)
    with lock:
        rows = read_manifest()
        removed = [r for r in rows if r["path"] in wanted]
        if not removed:
            raise ValueError("None of those frames exist")
        _check_not_recording(removed)
        action_id = uuid.uuid4().hex[:8]
        for r in removed:
            src, dst = DATA_DIR / r["path"], TRASH_DIR / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
            _append_csv(TRASH_ROWS, ["action_id"] + FIELDS, {"action_id": action_id, **r})
        _write_manifest([r for r in rows if r["path"] not in wanted])
        _log("trash", action_id, {r["session_id"] for r in removed}, len(removed), reason)
        log.info("Trashed %d frames (action %s): %s", len(removed), action_id, reason)
        return {"action_id": action_id, "frames": len(removed)}


def restore(action_id: str) -> int:
    """Undo a trash action: move its frames back and re-add their rows."""
    with lock:
        if not TRASH_ROWS.exists():
            raise ValueError("The trash is empty")
        with TRASH_ROWS.open(newline="") as f:
            trashed = list(csv.DictReader(f))
        back = [r for r in trashed if r["action_id"] == action_id]
        if not back:
            raise ValueError("Unknown trash action")
        rows = read_manifest()
        for r in back:
            src, dst = TRASH_DIR / r["path"], DATA_DIR / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
            rows.append({k: r[k] for k in FIELDS})
        _write_manifest(rows)
        with TRASH_ROWS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["action_id"] + FIELDS)
            writer.writeheader()
            writer.writerows([r for r in trashed if r["action_id"] != action_id])
        _log("restore", action_id, {r["session_id"] for r in back}, len(back), "undo")
        log.info("Restored %d frames (action %s)", len(back), action_id)
        return len(back)


def trash_actions() -> list:
    """Summarise what is in the trash, newest first."""
    with lock:
        if not TRASH_ROWS.exists():
            return []
        with TRASH_ROWS.open(newline="") as f:
            rows = list(csv.DictReader(f))
        reasons = {}
        if CURATION_LOG.exists():
            with CURATION_LOG.open(newline="") as f:
                for entry in csv.DictReader(f):
                    if entry["action"] == "trash":
                        reasons[entry["action_id"]] = (entry["timestamp"], entry["detail"])
    groups = {}
    for r in rows:
        g = groups.setdefault(r["action_id"], {"frames": 0, "sessions": set()})
        g["frames"] += 1
        g["sessions"].add(r["session_id"])
    out = []
    for action_id, g in groups.items():
        when, why = reasons.get(action_id, ("", ""))
        out.append({"action_id": action_id, "frames": g["frames"],
                    "sessions": sorted(g["sessions"]), "when": when, "reason": why})
    return sorted(out, key=lambda a: a["when"], reverse=True)


def relabel(paths: list, label: str) -> int:
    """Change the label of the given frames; their lighting stays as recorded."""
    wanted = set(paths)
    with lock:
        rows = read_manifest()
        targets = [r for r in rows if r["path"] in wanted]
        if not targets:
            raise ValueError("None of those frames exist")
        _check_not_recording(targets)
        for r in targets:
            validate_combo(label, r["lighting"])
        old = sorted({r["label"] for r in targets})
        for r in targets:
            r["label"] = label
        _write_manifest(rows)
        _log("relabel", uuid.uuid4().hex[:8], {r["session_id"] for r in targets},
             len(targets), f"{'/'.join(old)} -> {label}")
        log.info("Relabelled %d frames as %s", len(targets), label)
        return len(targets)


def set_lighting(paths: list, lighting: str) -> int:
    """Change the lighting tag of the given frames; their labels stay as they are."""
    wanted = set(paths)
    with lock:
        rows = read_manifest()
        targets = [r for r in rows if r["path"] in wanted]
        if not targets:
            raise ValueError("None of those frames exist")
        _check_not_recording(targets)
        for r in targets:
            validate_combo(r["label"], lighting)
        old = sorted({r["lighting"] for r in targets})
        for r in targets:
            r["lighting"] = lighting
        _write_manifest(rows)
        _log("relight", uuid.uuid4().hex[:8], {r["session_id"] for r in targets},
             len(targets), f"{'/'.join(old)} -> {lighting}")
        log.info("Changed lighting of %d frames to %s", len(targets), lighting)
        return len(targets)
