"""Run the DVC pipeline and record the run (the pipeline management entry point).

    python scripts/run_pipeline.py                      # dvc repro, everything that changed
    python scripts/run_pipeline.py --trigger drift --reason "RoomSenseDrift"
    python scripts/run_pipeline.py train export         # only up to these stages

`dvc repro` decides what has to run. This wrapper watches it and writes one
"pipeline run" to the MLflow experiment `roomsense-pipeline`:

  * status FINISHED / FAILED, the failed stage and the tail of the log
  * per stage: ran / cached / failed, seconds, items processed, throughput
  * tags for reproducibility: Git commit, DVC data version, trigger
  * artifacts: the DAG (from dvc.yaml), every reports/*.json, dvc.lock, the log

The Pipeline page of the web UI reads these runs through the API, so every
successful and failed run shows up in one console. Nothing here is needed for
the pipeline itself: if MLflow is unreachable the run is still written to
reports/pipeline_run.json and the exit code is dvc's.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import provenance, read_stage, setup_logging, utc_now  # noqa: E402

log = setup_logging("pipeline")
EXPERIMENT = os.environ.get("PIPELINE_EXPERIMENT", "roomsense-pipeline")
RUN_FILE = Path("reports/pipeline_run.json")
RUNNING = re.compile(r"Running stage '([^']+)'")
SKIPPED = re.compile(r"Stage '([^']+)' didn't change, skipping")
FAILED = re.compile(r"failed to reproduce '([^']+)'")


def _paths(entries) -> list:
    """dvc.yaml lists deps/outs as strings or one-key dicts ({path: {cache: false}})."""
    return [e if isinstance(e, str) else next(iter(e)) for e in (entries or [])]


def parse_dag(path: str = "dvc.yaml") -> dict:
    """Stages and edges of the DVC DAG. An edge a -> b exists when b depends on an output of a."""
    stages = []
    for name, spec in (yaml.safe_load(Path(path).read_text()) or {}).get("stages", {}).items():
        stages.append({"name": name, "cmd": spec.get("cmd", ""), "deps": _paths(spec.get("deps")),
                       "outs": _paths(spec.get("outs")) + _paths(spec.get("metrics")) + _paths(spec.get("plots")),
                       "params": _paths(spec.get("params"))})
    edges = []
    for a in stages:
        for b in stages:
            if a is not b and any(d == o or d.startswith(o + "/") for d in b["deps"] for o in a["outs"]):
                edges.append([a["name"], b["name"]])
    return {"stages": stages, "edges": edges}


def run_dvc(targets: list, names: list) -> dict:
    """Run `dvc repro`, echo its output, and work out what happened to every stage."""
    state = {n: {"status": "not_reached"} for n in names}
    lines, current, t_stage = [], None, None
    t0 = time.perf_counter()

    def close(status: str) -> None:
        if current and state[current]["status"] == "running":
            state[current].update(status=status, wall_seconds=round(time.perf_counter() - t_stage, 3))

    try:
        proc = subprocess.Popen(["dvc", "repro", *targets], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except OSError as e:
        return {"returncode": 127, "stages": state, "log": [f"could not start dvc: {e}"], "seconds": 0.0,
                "failed_stage": None}
    failed = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line.rstrip("\n"))
        if m := RUNNING.search(line):
            close("ok")
            current, t_stage = m.group(1), time.perf_counter()
            state.setdefault(current, {})["status"] = "running"
        elif m := SKIPPED.search(line):
            state.setdefault(m.group(1), {})["status"] = "cached"
        elif m := FAILED.search(line):
            failed = m.group(1)
    rc = proc.wait()
    if rc == 0:
        close("ok")
    else:
        failed = failed or current
        close("failed")
        if failed:
            state.setdefault(failed, {})["status"] = "failed"
    return {"returncode": rc, "stages": state, "log": lines[-400:], "seconds": round(time.perf_counter() - t0, 3),
            "failed_stage": failed if rc else None}


def merge_stage_records(result: dict) -> None:
    """Add what each stage script measured itself (reports/stages/*.json)."""
    for name, st in result["stages"].items():
        rec = read_stage(name)
        if st["status"] in ("ok", "cached") and rec.get("status") == "ok":
            st["seconds"] = rec.get("duration_s")
            for k in ("items", "unit", "throughput_per_s", "finished_at"):
                if k in rec:
                    st[k] = rec[k]
        elif st["status"] == "failed":
            st["seconds"] = rec.get("duration_s", st.get("wall_seconds"))
            st["error"] = rec.get("error") if rec.get("status") == "failed" else None
        if "seconds" not in st and "wall_seconds" in st:
            st["seconds"] = st["wall_seconds"]


def headline() -> dict:
    """A few numbers from the stage reports for the run overview."""
    def load(name):
        try:
            return json.loads(Path(f"reports/{name}").read_text())
        except (OSError, ValueError):
            return {}
    prep, pre, met, exp, reg = (load(n) for n in ("prepare_report.json", "preprocess_report.json", "metrics.json",
                                                   "export_report.json", "register_report.json"))
    out = {
        "frames_raw": prep.get("frames_in"), "frames_valid": prep.get("frames_kept"),
        "frames_kept": pre.get("kept"), "sessions": pre.get("sessions"),
        "ray_resize_fps": (pre.get("throughput_fps") or {}).get("ray_resize"),
        "preprocess_fps": (pre.get("throughput_fps") or {}).get("end_to_end"),
        "train_images_per_s": met.get("train_images_per_s"),
        "test_macro_f1": exp.get("onnx_test_macro_f1", met.get("test_macro_f1")),
        "latency_ms_p95": exp.get("latency_ms_p95"),
        "model_version": reg.get("version"), "promoted": reg.get("promoted"),
    }
    return {k: v for k, v in out.items() if v is not None}


def publish(result: dict) -> str:
    """Write the pipeline run to MLflow. Returns the run ID ("" when MLflow is unreachable)."""
    try:
        import mlflow
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"pipeline-{result['trigger']}") as run:
            tags = {"trigger": result["trigger"], "reason": result["reason"][:250], "host": result["host"],
                    "git_commit": result["git_commit"], "git_dirty": str(result["git_dirty"]).lower(),
                    "data_version": result["data_version"], "result": result["status"],
                    "started_at": result["started_at"], "finished_at": result["finished_at"]}
            metrics = {"total_seconds": result["seconds"]}
            for name, st in result["stages"].items():
                tags[f"stage_{name}_status"] = st["status"]
                for key, metric in (("seconds", "seconds"), ("items", "items"), ("throughput_per_s", "throughput")):
                    if isinstance(st.get(key), (int, float)):
                        metrics[f"stage_{name}_{metric}"] = float(st[key])
            for status in ("ok", "cached", "failed"):
                metrics[f"stages_{status}"] = sum(1 for st in result["stages"].values() if st["status"] == status)
            for key, value in result["headline"].items():
                if isinstance(value, bool):
                    tags[key] = str(value).lower()
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
                else:
                    tags[key] = str(value)
            if result["failed_stage"]:
                tags["failed_stage"] = result["failed_stage"]
                tags["error"] = (result["stages"].get(result["failed_stage"], {}).get("error")
                                 or "\n".join(result["log"][-8:]))[:1000]
            mlflow.set_tags(tags)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(result["dag"], "pipeline/dag.json")
            mlflow.log_dict({k: v for k, v in result.items() if k != "log"}, "pipeline/pipeline_run.json")
            mlflow.log_text("\n".join(result["log"]), "pipeline/dvc_repro.log")
            for extra in [*sorted(Path("reports").glob("*.json")), Path("dvc.lock"), Path("params.yaml")]:
                if extra.exists():
                    mlflow.log_artifact(str(extra), artifact_path="pipeline")
            run_id = run.info.run_id
            # Leave the `with` block cleanly, then mark a failed pipeline as FAILED.
        if result["status"] != "ok":
            mlflow.MlflowClient().set_terminated(run_id, status="FAILED")
        return run_id
    except Exception as e:  # noqa: BLE001  (tracking must never break the pipeline itself)
        log.warning("pipeline run not recorded in MLflow: %s", e)
        return ""


def run_pipeline(trigger: str = "manual", reason: str = "", targets=None) -> dict:
    """Run the pipeline once and return the full run record."""
    dag = parse_dag()
    started = utc_now()
    log.info("pipeline started (trigger=%s%s)", trigger, f", reason={reason}" if reason else "")
    result = run_dvc(list(targets or []), [s["name"] for s in dag["stages"]])
    merge_stage_records(result)
    result.update(provenance())
    result.update(trigger=trigger, reason=reason, host=socket.gethostname(), started_at=started,
                  finished_at=utc_now(), status="ok" if result["returncode"] == 0 else "failed",
                  dag=dag, headline=headline())
    result["mlflow_run_id"] = publish(result)
    try:
        RUN_FILE.parent.mkdir(exist_ok=True)
        RUN_FILE.write_text(json.dumps({k: v for k, v in result.items() if k != "log"}, indent=2))
    except OSError as e:
        log.warning("could not write %s: %s", RUN_FILE, e)
    if result["status"] == "ok":
        log.info("pipeline finished in %.1f s", result["seconds"])
    else:
        log.error("pipeline FAILED at stage %s after %.1f s", result["failed_stage"], result["seconds"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the DVC pipeline and record the run in MLflow.")
    ap.add_argument("targets", nargs="*", help="optional DVC stage names (default: everything)")
    ap.add_argument("--trigger", default="manual", help="manual | drift | retrain | mlproject | demo")
    ap.add_argument("--reason", default="", help="free text shown in the pipeline console")
    a = ap.parse_args()
    return run_pipeline(a.trigger, a.reason, a.targets)["returncode"]


if __name__ == "__main__":
    sys.exit(main())
