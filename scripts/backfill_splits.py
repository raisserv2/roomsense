"""Attach a model's training split to its MLflow run, for models registered
before the register stage started recording it.

The promotion gate compares a candidate with the champion only on test
sessions the champion never learned from, so it needs the champion's
splits.csv (stored as `pipeline/splits.csv` in the model version's run). For
an older model, recover that file from the DVC cache using the dvc.lock of
the commit that trained it:

    python scripts/backfill_splits.py --version 2 --lock-rev <commit>
    python scripts/backfill_splits.py --version 2 --splits path/to/splits.csv

Needs MLFLOW_TRACKING_URI (https://nova.tailb5cb87.ts.net:5443 on the Legion).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import setup_logging  # noqa: E402

log = setup_logging("backfill-splits")
SPLITS = "data/processed/splits.csv"


def splits_md5(lock: dict) -> str:
    """md5 of data/processed/splits.csv in a parsed dvc.lock (raises KeyError if absent)."""
    for out in (lock.get("stages", {}).get("split", {}) or {}).get("outs", []):
        if out.get("path") == SPLITS:
            return out["md5"]
    raise KeyError(f"{SPLITS} is not an output of the split stage in this dvc.lock")


def cache_path(md5: str, cache: Path = Path(".dvc/cache")):
    """Where DVC 3 (files/md5/xx/...) or DVC 2 (xx/...) keeps that file, or None."""
    for p in (cache / "files" / "md5" / md5[:2] / md5[2:], cache / md5[:2] / md5[2:]):
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Attach a model version's splits.csv to its MLflow run.")
    ap.add_argument("--version", required=True, help="registry version, e.g. 2")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lock-rev", help="git commit whose dvc.lock trained that version")
    src.add_argument("--splits", help="path to the splits.csv directly")
    a = ap.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.error("MLFLOW_TRACKING_URI is not set")
        return 2
    if a.splits:
        path = Path(a.splits)
    else:
        try:
            lock = yaml.safe_load(subprocess.check_output(["git", "show", f"{a.lock_rev}:dvc.lock"], text=True))
            path = cache_path(splits_md5(lock))
        except (subprocess.CalledProcessError, KeyError) as e:
            log.error("could not read the split from %s: %s", a.lock_rev, e)
            return 1
        if path is None:
            log.error("that splits.csv is not in the local DVC cache (try `dvc fetch` on the commit)")
            return 1

    import tempfile

    import pandas as pd
    from mlflow import MlflowClient

    df = pd.read_csv(path)
    if not {"session_id", "split"} <= set(df.columns):
        log.error("%s has no session_id/split columns", path)
        return 1
    params = (yaml.safe_load(Path("params.yaml").read_text()) or {}).get("register") or {}
    client = MlflowClient()
    run_id = client.get_model_version(params.get("model_name", "roomsense"), a.version).run_id
    # DVC cache files are named by hash, so upload a copy named splits.csv
    with tempfile.TemporaryDirectory() as tmp:
        named = Path(tmp) / "splits.csv"
        named.write_bytes(path.read_bytes())
        client.log_artifact(run_id, str(named), artifact_path="pipeline")
    counts = df.groupby("split").session_id.nunique().to_dict()
    log.info("v%s (run %s): stored pipeline/splits.csv, sessions per split %s", a.version, run_id[:8], counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
