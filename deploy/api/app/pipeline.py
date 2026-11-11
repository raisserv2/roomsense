"""Pipeline console: what the Pipeline page of the web UI shows.

The training pipeline runs on another machine (the Legion). Everything the UI
needs to know about it is read from MLflow, the one place both machines share:

  * experiment `roomsense-pipeline`: one run per pipeline execution, written by
    scripts/run_pipeline.py (status, per-stage timings and throughput, the DAG)
  * experiment `roomsense`: the training runs (metrics, Git commit, data version)
  * the model registry: versions and the `champion` alias

`PipelineStore` keeps a cached snapshot that a background thread refreshes, so
the HTTP endpoints stay fast and keep answering (with the last snapshot and an
error note) while MLflow is down.
"""
import logging
import os
import threading
import time

import requests

from roomsense import registry

from . import metrics as M
from . import settings as S

log = logging.getLogger("pipeline")

# What each stage does, in plain words (shown under the DAG).
STAGE_INFO = {
    "capture": {"title": "Capture", "tool": "FastAPI on nova", "machine": "nova",
                "what": "Saves a labelled camera frame every 5 s while a recording session runs."},
    "prepare": {"title": "Prepare", "tool": "DVC", "machine": "Legion",
                "what": "Checks every row of the frame list (file, label, lighting, timestamp) and adds day/night."},
    "preprocess": {"title": "Preprocess", "tool": "Ray", "machine": "Legion",
                   "what": "Trims session edges, resizes to 224x224 and drops near-duplicate frames, in parallel."},
    "split": {"title": "Split", "tool": "DVC", "machine": "Legion",
              "what": "Assigns whole sessions to train, validation and test so no session leaks."},
    "train": {"title": "Train", "tool": "PyTorch + MLflow", "machine": "Legion",
              "what": "Fine-tunes MobileNetV3-Small on the GPU and calibrates its confidence."},
    "export": {"title": "Export", "tool": "ONNX", "machine": "Legion",
               "what": "Converts the model for CPU serving, checks it matches PyTorch, measures latency."},
    "register": {"title": "Register", "tool": "MLflow registry", "machine": "Legion",
                 "what": "Stores the model and promotes it to champion only if it beats the current one."},
    "serve": {"title": "Serve", "tool": "FastAPI + ONNX Runtime", "machine": "nova",
              "what": "Loads the champion and classifies the live camera every 5 s."},
}

# Fallback DAG (mirrors dvc.yaml) until the first recorded pipeline run supplies the real one.
STATIC_DAG = {
    "stages": [{"name": n} for n in ("prepare", "preprocess", "split", "train", "export", "register")],
    "edges": [["prepare", "preprocess"], ["preprocess", "split"], ["preprocess", "train"], ["split", "train"],
              ["train", "export"], ["split", "export"], ["preprocess", "export"], ["export", "register"]],
}


def _ms(value):
    return value / 1000 if value else None


def pipeline_run_view(run) -> dict:
    """An MLflow run of the pipeline experiment as a plain dict for the UI."""
    tags, metrics = run.data.tags, run.data.metrics
    stages = {}
    for key, value in tags.items():
        if key.startswith("stage_") and key.endswith("_status"):
            name = key[len("stage_"):-len("_status")]
            stages[name] = {"status": value, "seconds": metrics.get(f"stage_{name}_seconds"),
                            "items": metrics.get(f"stage_{name}_items"),
                            "throughput_per_s": metrics.get(f"stage_{name}_throughput")}
    start, end = _ms(run.info.start_time), _ms(run.info.end_time)
    headline = {k: metrics[k] for k in ("frames_raw", "frames_valid", "frames_kept", "sessions", "ray_resize_fps",
                                        "preprocess_fps", "train_images_per_s", "test_macro_f1", "latency_ms_p95")
                if k in metrics}
    return {
        "run_id": run.info.run_id, "status": run.info.status, "ok": run.info.status == "FINISHED",
        "started_at": start, "seconds": metrics.get("total_seconds", (end - start) if start and end else None),
        "trigger": tags.get("trigger", ""), "reason": tags.get("reason", ""), "host": tags.get("host", ""),
        "git_commit": tags.get("git_commit", ""), "git_dirty": tags.get("git_dirty") == "true",
        "data_version": tags.get("data_version", ""),
        "failed_stage": tags.get("failed_stage"), "error": tags.get("error"),
        "model_version": tags.get("model_version"), "promoted": tags.get("promoted"),
        "stages": stages, "headline": headline,
    }


def training_run_view(run) -> dict:
    """An MLflow training run as a plain dict for the UI."""
    tags, metrics, params = run.data.tags, run.data.metrics, run.data.params
    start, end = _ms(run.info.start_time), _ms(run.info.end_time)
    return {
        "run_id": run.info.run_id, "name": tags.get("mlflow.runName", ""), "status": run.info.status,
        "started_at": start, "seconds": (end - start) if start and end else None,
        "git_commit": tags.get("git_commit") or tags.get("mlflow.source.git.commit", ""),
        "data_version": tags.get("data_version", ""),
        "test_macro_f1": metrics.get("test_macro_f1"), "test_accuracy": metrics.get("test_accuracy"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
        "train_images_per_s": metrics.get("train_images_per_s"),
        "train_frames": params.get("train_frames"), "classes": params.get("classes"),
    }


class PipelineStore:
    """Cached view of pipeline runs, training runs, the registry and data ingestion."""

    def __init__(self, client_factory=None, stats_fetcher=None):
        self._client_factory = client_factory or self._default_client
        self._stats_fetcher = stats_fetcher or self._fetch_capture_stats
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._snap = None
        self._dag_cache = (None, None)  # (run_id, dag)

    # --- data sources -------------------------------------------------------
    @staticmethod
    def _default_client():
        """An MlflowClient, or None when no tracking server is configured."""
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return None
        from mlflow import MlflowClient
        return MlflowClient()

    @staticmethod
    def _fetch_capture_stats() -> dict:
        if not S.CAPTURE_STATS_URL:
            return {"available": False, "error": "capture service not configured"}
        try:
            r = requests.get(S.CAPTURE_STATS_URL, timeout=4)
            r.raise_for_status()
            return {"available": True, **r.json()}
        except (requests.RequestException, ValueError) as e:
            return {"available": False, "error": str(e)[:200]}

    @staticmethod
    def _runs(client, experiment: str, limit: int) -> list:
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return []
        return list(client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"],
                                       max_results=limit))

    def _dag(self, latest_run_id) -> dict:
        """The DAG stored with the latest pipeline run (parsed from dvc.yaml), else the static copy."""
        if not latest_run_id:
            return STATIC_DAG
        if self._dag_cache[0] == latest_run_id:
            return self._dag_cache[1]
        dag = STATIC_DAG
        try:
            import mlflow
            loaded = mlflow.artifacts.load_dict(f"runs:/{latest_run_id}/pipeline/dag.json")
            if loaded.get("stages"):
                dag = {"stages": [{"name": s["name"], "cmd": s.get("cmd", ""), "deps": s.get("deps", []),
                                   "outs": s.get("outs", [])} for s in loaded["stages"]],
                       "edges": loaded.get("edges", [])}
        except Exception as e:  # noqa: BLE001  (artifact missing or MLflow busy: the static DAG is fine)
            log.debug("no DAG artifact for run %s: %s", latest_run_id, e)
        self._dag_cache = (latest_run_id, dag)
        return dag

    # --- snapshot -------------------------------------------------------------
    def refresh(self) -> dict:
        """Rebuild the snapshot from MLflow and the capture service. Never raises."""
        with self._refresh_lock:
            errors, runs, training, versions, mlflow_up = [], [], [], [], False
            try:
                client = self._client_factory()
                if client is None:
                    errors.append("MLflow is not configured (MLFLOW_TRACKING_URI is empty).")
                else:
                    runs = [pipeline_run_view(r) for r in self._runs(client, S.PIPELINE_EXPERIMENT, 50)]
                    training = [training_run_view(r) for r in self._runs(client, S.TRAIN_EXPERIMENT, 50)]
                    versions = registry.list_versions(client, S.MODEL_NAME, S.MODEL_ALIAS)
                    mlflow_up = True
            except Exception as e:  # noqa: BLE001  (MLflow down or slow: keep serving the last snapshot)
                log.warning("pipeline refresh failed: %s", e)
                errors.append(f"MLflow is not answering: {str(e)[:200]}")
            M.DEPENDENCY_UP.labels("mlflow").set(1 if mlflow_up else 0)

            previous = self._snap or {}
            if not mlflow_up and previous:
                runs, training, versions = previous["runs"], previous["training_runs"], previous["versions"]
            latest = runs[0] if runs else None
            last_ok = next((r for r in runs if r["ok"]), None)
            ok, failed = sum(1 for r in runs if r["ok"]), sum(1 for r in runs if r["status"] == "FAILED")
            champion = next((v["version"] for v in versions if v["is_champion"]), None)
            snap = {
                "generated_at": time.time(),
                "mlflow_up": mlflow_up,
                "errors": errors,
                "dag": self._dag(latest["run_id"] if latest and mlflow_up else None),
                "stage_info": STAGE_INFO,
                "latest": latest,
                "last_success": last_ok,
                "summary": {
                    "runs_total": len(runs), "runs_ok": ok, "runs_failed": failed,
                    "success_rate": round(ok / (ok + failed), 3) if ok + failed else None,
                    "last_success_at": last_ok["started_at"] if last_ok else None,
                    "last_failure_at": next((r["started_at"] for r in runs if r["status"] == "FAILED"), None),
                    "training_runs": len(training), "model_versions": len(versions), "champion": champion,
                },
                "ingestion": self._stats_fetcher(),
                "runs": runs,
                "training_runs": training,
                "versions": versions,
            }
            self._export_metrics(snap)
            with self._lock:
                self._snap = snap
            return snap

    @staticmethod
    def _export_metrics(snap: dict) -> None:
        """Mirror the latest pipeline run into Prometheus gauges (for Grafana and alerts)."""
        s = snap["summary"]
        M.PIPE_RUNS.labels("ok").set(s["runs_ok"])
        M.PIPE_RUNS.labels("failed").set(s["runs_failed"])
        M.REGISTRY_VERSIONS.set(s["model_versions"])
        latest = snap["latest"]
        if not latest:
            return
        M.PIPE_LAST_OK.set(1 if latest["ok"] else 0)
        M.PIPE_LAST_TS.set(latest["started_at"] or 0)
        M.PIPE_SECONDS.set(latest["seconds"] or 0)
        source = snap["last_success"] or latest
        for name, st in source["stages"].items():
            if st.get("seconds") is not None:
                M.STAGE_SECONDS.labels(name).set(st["seconds"])
            if st.get("throughput_per_s") is not None:
                M.STAGE_THROUGHPUT.labels(name).set(st["throughput_per_s"])
        if source["headline"].get("ray_resize_fps") is not None:
            M.RAY_FPS.set(source["headline"]["ray_resize_fps"])

    def snapshot(self, max_age: float = None) -> dict:
        """The cached snapshot; refreshed first when missing or older than `max_age` seconds."""
        with self._lock:
            snap = self._snap
        if snap is None or (max_age is not None and time.time() - snap["generated_at"] > max_age):
            snap = self.refresh()
        return snap

    def poll_forever(self, interval: float, stop: threading.Event) -> None:
        """Background thread: keep the snapshot fresh."""
        while not stop.is_set():
            self.refresh()
            stop.wait(interval)
