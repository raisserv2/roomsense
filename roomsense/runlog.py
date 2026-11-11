"""Logging, stage timing and provenance shared by every pipeline script.

Every DVC stage calls ``start_stage("<name>")`` first and ``.done(...)`` last.
That gives all scripts the same log format and leaves a small JSON record in
``reports/stages/<name>.json`` (status, duration, items processed, throughput).
``scripts/run_pipeline.py`` collects those records into one MLflow "pipeline
run", which is what the Pipeline page of the web UI shows.
"""
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
STAGE_DIR = Path("reports/stages")


def setup_logging(name: str, level: str = "") -> logging.Logger:
    """Configure the root logger once (level from $LOG_LEVEL) and return a named logger."""
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
    return logging.getLogger(name)


def _git(*args: str) -> str:
    """Output of a git command, or "" when git or the repository is not available."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_commit(short: bool = False) -> str:
    """Hash of the checked-out commit ("" outside a Git repository)."""
    return _git("rev-parse", "--short", "HEAD") if short else _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    """True when tracked files have uncommitted changes (the run is then not exactly reproducible)."""
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def data_version(dvc_file: str = "data/raw.dvc") -> str:
    """The dataset fingerprint (md5) DVC pinned in ``data/raw.dvc``, or ""."""
    try:
        for line in Path(dvc_file).read_text().splitlines():
            if line.strip().startswith("- md5:") or line.strip().startswith("md5:"):
                return line.split("md5:", 1)[1].strip()
    except OSError:
        pass
    return ""


def provenance() -> dict:
    """What is needed to reproduce a run: code version + data version."""
    return {"git_commit": git_commit(), "git_dirty": git_dirty(), "data_version": data_version()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageRun:
    """Times one pipeline stage and writes its record to ``reports/stages``."""

    def __init__(self, name: str, log: logging.Logger, out_dir: Path = STAGE_DIR):
        self.name, self.log = name, log
        self.path = Path(out_dir) / f"{name}.json"
        self._t0 = time.perf_counter()
        self.record = {"stage": name, "status": "running", "started_at": utc_now(), "git_commit": git_commit()}
        self._write()
        log.info("stage %s started", name)

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.record, indent=2))
        except OSError as e:  # a full disk must not hide the real result of the stage
            self.log.warning("could not write %s: %s", self.path, e)

    def _finish(self, status: str) -> None:
        duration = time.perf_counter() - self._t0
        self.record.update(status=status, finished_at=utc_now(), duration_s=round(duration, 3))

    def done(self, items=None, unit: str = "frames", **extra) -> dict:
        """Mark the stage as finished. ``items`` (frames, usually) gives the throughput."""
        self._finish("ok")
        if items is not None:
            d = max(self.record["duration_s"], 1e-9)
            self.record.update(items=int(items), unit=unit, throughput_per_s=round(items / d, 3))
        self.record.update(extra)
        self._write()
        self.log.info("stage %s finished in %.1f s%s", self.name, self.record["duration_s"],
                      f" ({self.record['throughput_per_s']:.1f} {unit}/s)" if items is not None else "")
        return self.record

    def fail(self, error: str) -> dict:
        self._finish("failed")
        self.record["error"] = str(error)[:2000]
        self._write()
        self.log.error("stage %s failed after %.1f s: %s", self.name, self.record["duration_s"], error)
        return self.record


def start_stage(name: str) -> StageRun:
    """Set up logging, start the timer, and record a failure if the script dies."""
    log = setup_logging(name)
    run = StageRun(name, log)
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if run.record["status"] == "running":
            run.fail(f"{exc_type.__name__}: {exc}")
        previous(exc_type, exc, tb)

    sys.excepthook = hook
    return run


def read_stage(name: str, out_dir: Path = STAGE_DIR) -> dict:
    """The last record of a stage ({} when it never ran)."""
    try:
        return json.loads((Path(out_dir) / f"{name}.json").read_text())
    except (OSError, ValueError):
        return {}
