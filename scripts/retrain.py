"""Drift -> retrain loop, run on the Legion (it has the GPU).

The API on nova receives Prometheus/Alertmanager drift alerts and raises a
"retrain requested" flag. This script checks that flag and, when it is set:

  git pull  ->  dvc pull  ->  pipeline (dvc repro)  ->  dvc push  ->  git commit + push  ->  ack

`dvc repro` reruns only what changed (new nightly frames, new params), and the
`register` stage promotes the new model if it beats the champion. nova's API
then hot-reloads it, and the drift signal drops back once the new model's
reference statistics include the new conditions. A failed pipeline is recorded
(MLflow experiment `roomsense-pipeline`) and the flag stays raised.

Usage:
  python scripts/retrain.py            # check once, retrain if requested
  python scripts/retrain.py --force    # retrain now regardless of the flag
  python scripts/retrain.py --watch 10 # check every 10 minutes
  python scripts/retrain.py --no-git   # skip git pull/commit/push
Env: ROOMSENSE_API (default https://nova.tailb5cb87.ts.net:4443/api), RETRAIN_TOKEN
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import setup_logging  # noqa: E402
from scripts.run_pipeline import run_pipeline  # noqa: E402

log = setup_logging("retrain")
API = os.environ.get("ROOMSENSE_API", "https://nova.tailb5cb87.ts.net:4443/api").rstrip("/")
COMMIT_FILES = ["dvc.lock", "params.yaml", "reports/metrics.json", "reports/split_report.json",
                "reports/prepare_report.json", "reports/preprocess_report.json", "reports/export_report.json",
                "reports/register_report.json", "reports/pipeline_run.json", "reports/stages"]


class PipelineError(RuntimeError):
    """The DVC pipeline exited with an error."""


def sh(*cmd: str) -> None:
    """Run a command, echoing it first. Raises CalledProcessError on a non-zero exit."""
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def status() -> dict:
    """The retrain flag as the API reports it."""
    r = requests.get(f"{API}/retrain", timeout=10)
    r.raise_for_status()
    return r.json()


def retrain(reason: str, use_git: bool, trigger: str = "drift") -> None:
    started = datetime.now().isoformat(timespec="seconds")
    if use_git:
        sh("git", "pull", "--ff-only")
    sh("dvc", "pull")
    result = run_pipeline(trigger=trigger, reason=reason)
    if result["returncode"] != 0:
        raise PipelineError(f"stage {result['failed_stage']} failed")
    sh("dvc", "push")
    report = Path("reports/register_report.json")
    reg = json.loads(report.read_text()) if report.exists() else {}
    if use_git:
        files = [f for f in COMMIT_FILES if Path(f).exists()]
        sh("git", "add", *files)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0:
            sh("git", "commit", "-m", f"retrain: {reason} (model v{reg.get('version', '?')})")
            sh("git", "push")
    try:
        requests.post(f"{API}/retrain/ack", timeout=10, json={
            "started": started, "reason": reason, "version": reg.get("version"),
            "promoted": reg.get("promoted"), "detail": reg.get("reason"),
            "pipeline_run_id": result.get("mlflow_run_id"), "seconds": result.get("seconds")},
            headers={"X-Token": os.environ.get("RETRAIN_TOKEN", "")}).raise_for_status()
    except requests.RequestException as e:
        log.warning("Could not acknowledge on the API: %s", e)
    log.info("retrain done: %s", json.dumps(reg))


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain when the API on nova has raised the retrain flag.")
    ap.add_argument("--force", action="store_true", help="retrain now, whatever the flag says")
    ap.add_argument("--watch", type=float, help="poll interval in minutes")
    ap.add_argument("--no-git", action="store_true", help="skip git pull/commit/push")
    a = ap.parse_args()
    while True:
        try:
            s = {"requested": True, "reason": "manual --force"} if a.force else status()
            if s.get("requested"):
                retrain(s.get("reason") or "drift alert", not a.no_git, "manual" if a.force else "drift")
                a.force = False
            else:
                log.info("no retrain requested")
        except (requests.RequestException, subprocess.CalledProcessError, PipelineError) as e:
            log.error("retrain check failed: %s", e)
            if not a.watch:
                return 1
        if not a.watch:
            return 0
        time.sleep(a.watch * 60)


if __name__ == "__main__":
    sys.exit(main())
