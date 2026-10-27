"""Drift -> retrain loop, run on the Legion (it has the GPU).

The API on nova receives Prometheus/Alertmanager drift alerts and raises a
"retrain requested" flag. This script checks that flag and, when it is set:

  git pull  ->  dvc pull  ->  dvc repro  ->  dvc push  ->  git commit + push  ->  ack

`dvc repro` reruns only what changed (new nightly frames, new params), and the
`register` stage promotes the new model if it beats the champion. nova's API
then hot-reloads it, and the drift PSI drops back once the new model's
reference statistics include the new conditions.

Usage:
  python scripts/retrain.py            # check once, retrain if requested
  python scripts/retrain.py --force    # retrain now regardless of the flag
  python scripts/retrain.py --watch 10 # check every 10 minutes
  python scripts/retrain.py --no-git   # skip git pull/commit/push
Env: ROOMSENSE_API (default https://nova.tailb5cb87.ts.net:4443/api)
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

API = os.environ.get("ROOMSENSE_API", "https://nova.tailb5cb87.ts.net:4443/api").rstrip("/")


def sh(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def status() -> dict:
    r = requests.get(f"{API}/retrain", timeout=10)
    r.raise_for_status()
    return r.json()


def retrain(reason: str, use_git: bool) -> None:
    started = datetime.now().isoformat(timespec="seconds")
    if use_git:
        sh("git", "pull", "--ff-only")
    sh("dvc", "pull")
    sh("dvc", "repro")
    sh("dvc", "push")
    reg = json.loads(Path("reports/register_report.json").read_text()) if Path("reports/register_report.json").exists() else {}
    if use_git:
        files = [f for f in ["dvc.lock", "params.yaml", "reports/metrics.json", "reports/split_report.json",
                             "reports/preprocess_report.json", "reports/export_report.json",
                             "reports/register_report.json"] if Path(f).exists()]
        sh("git", "add", *files)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
            sh("git", "commit", "-m", f"retrain: {reason} (model v{reg.get('version', '?')})")
            sh("git", "push")
    try:
        requests.post(f"{API}/retrain/ack", timeout=10, json={
            "started": started, "reason": reason, "version": reg.get("version"),
            "promoted": reg.get("promoted"), "detail": reg.get("reason")},
            headers={"X-Token": os.environ.get("RETRAIN_TOKEN", "")}).raise_for_status()
    except requests.RequestException as e:
        print(f"Could not acknowledge on the API: {e}")
    print(json.dumps(reg, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--watch", type=float, help="poll interval in minutes")
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            s = {"requested": True, "reason": "manual --force"} if a.force else status()
            if s.get("requested"):
                retrain(s.get("reason") or "drift alert", not a.no_git)
                a.force = False
            else:
                print(f"{datetime.now():%H:%M:%S} no retrain requested")
        except (requests.RequestException, subprocess.CalledProcessError) as e:
            print(f"retrain check failed: {e}")
            if not a.watch:
                return 1
        if not a.watch:
            return 0
        time.sleep(a.watch * 60)


if __name__ == "__main__":
    sys.exit(main())
