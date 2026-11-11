"""Small persistent state: the retrain request flag and the activity log (both survive restarts)."""
import json
import threading
import time
from pathlib import Path

from . import metrics as M


class RetrainState:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.data = {"requested": False, "reason": None, "requested_at": None, "alerts": 0, "history": []}
        try:
            self.data.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
        M.RETRAIN_PENDING.set(1 if self.data["requested"] else 0)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            tmp.replace(self.path)
        except OSError:
            pass

    def request(self, reason: str) -> bool:
        """Raise the flag. Returns False if it was already raised."""
        with self._lock:
            self.data["alerts"] += 1
            if self.data["requested"]:
                self._save()
                return False
            self.data.update(requested=True, reason=reason, requested_at=time.time())
            self._save()
        M.RETRAIN_REQUESTS.inc()
        M.RETRAIN_PENDING.set(1)
        return True

    def ack(self, detail: dict) -> None:
        with self._lock:
            entry = {"requested_at": self.data.get("requested_at"), "reason": self.data.get("reason"),
                     "done_at": time.time(), **detail}
            self.data["history"] = (self.data.get("history") or [])[-19:] + [entry]
            self.data.update(requested=False, reason=None, requested_at=None)
            self._save()
        M.RETRAINS_DONE.inc()
        M.RETRAIN_PENDING.set(0)

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.data))


class EventLog:
    """A short, persistent activity log: alerts received, retrains, model loads, failures.

    It backs the "Activity" console of the Pipeline page. Repeating the same
    event (a model load that fails every minute) only bumps a counter.
    """

    def __init__(self, path: Path, size: int = 200):
        self.path, self.size = path, size
        self._lock = threading.Lock()
        try:
            self.events = list(json.loads(path.read_text()))[-size:]
        except (OSError, ValueError):
            self.events = []

    def add(self, kind: str, message: str, level: str = "info", **detail) -> None:
        """level: info | success | warning | error."""
        now = time.time()
        with self._lock:
            last = self.events[-1] if self.events else None
            if last and last["kind"] == kind and last["message"] == message:
                last["count"] = last.get("count", 1) + 1
                last["last_at"] = now
            else:
                self.events.append({"at": now, "last_at": now, "kind": kind, "level": level,
                                    "message": message, "count": 1, **detail})
                self.events = self.events[-self.size:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.events))
                tmp.replace(self.path)
            except OSError:
                pass  # the log is a convenience; never fail a request because the disk is full

    def recent(self, limit: int = 50) -> list:
        with self._lock:
            return [dict(e) for e in reversed(self.events[-limit:])]
