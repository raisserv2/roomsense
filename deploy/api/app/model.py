"""Loads the serving model from the MLflow registry (or a local folder) and
hot-reloads it when the registry alias points to a new version."""
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from roomsense import frames
from roomsense.model_pkg import load_package

from . import metrics as M
from . import settings as S

log = logging.getLogger("model")


class Model:
    def __init__(self, onnx_path: Path, meta: dict, version: str, run_id: str, uri: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = S.ORT_THREADS
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.meta = meta
        self.classes = list(meta["classes"])
        self.size = int(meta.get("img_size", 224))
        self.reference = meta.get("reference") or {}
        self.version, self.run_id, self.uri = version, run_id, uri
        self.loaded_at = time.time()

    def predict(self, img_bgr: np.ndarray) -> dict:
        x = frames.to_input(img_bgr, self.size)
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: x})[0][0]
        latency = time.perf_counter() - t0
        p = frames.softmax(logits)
        i = int(p.argmax())
        return {"label": self.classes[i], "confidence": float(p[i]),
                "probabilities": {c: float(v) for c, v in zip(self.classes, p)},
                "latency_ms": latency * 1000, "_latency_s": latency}

    def info(self) -> dict:
        return {"version": self.version, "run_id": self.run_id, "uri": self.uri, "classes": self.classes,
                "metrics": self.meta.get("metrics"), "data_include": self.meta.get("data_include"),
                "created_at": self.meta.get("created_at"), "train_frames": self.meta.get("train_frames"),
                "reference_brightness_mean": self.reference.get("brightness_mean"),
                "loaded_at": self.loaded_at}


class ModelManager:
    """Holds the current model; swaps it atomically on reload."""

    def __init__(self, uri: str):
        self.uri = uri
        self.model = None
        self.error = None
        self._lock = threading.Lock()
        self._dir = Path(tempfile.mkdtemp(prefix="roomsense-model-"))

    # --- registry helpers -------------------------------------------------
    def _registry_target(self):
        """(version, run_id) the alias currently points to, or None for local paths."""
        if not self.uri.startswith("models:/"):
            return None
        from mlflow import MlflowClient
        ref = self.uri[len("models:/"):]
        client = MlflowClient()
        if "@" in ref:
            name, alias = ref.split("@", 1)
            mv = client.get_model_version_by_alias(name, alias)
        else:
            name, ver = ref.split("/", 1)
            mv = client.get_model_version(name, ver)
        return str(mv.version), mv.run_id or ""

    def load(self, force: bool = False) -> bool:
        """Load (or reload) the model if it changed. Returns True when a new model was loaded."""
        try:
            target = self._registry_target()
            if target and self.model and not force and target[0] == self.model.version:
                return False
            if target:
                import mlflow
                name = self.uri[len("models:/"):].split("@")[0].split("/")[0]
                dst = self._dir / f"v{target[0]}"
                if dst.exists():
                    shutil.rmtree(dst)
                path = mlflow.artifacts.download_artifacts(f"models:/{name}/{target[0]}", dst_path=str(dst))
                version, run_id = target
            else:
                path, version, run_id = self.uri, "local", ""
            onnx_path, meta = load_package(path)
            new = Model(onnx_path, meta, version, run_id, self.uri)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            M.RELOADS.labels("error").inc()
            log.warning("Model load failed: %s", e)
            return False
        with self._lock:
            old, self.model, self.error = self.model, new, None
        if old:
            M.MODEL_INFO.remove(old.version, old.run_id, old.uri)
        M.MODEL_INFO.labels(new.version, new.run_id, new.uri).set(1)
        M.MODEL_VERSION.set(float(new.version) if new.version.isdigit() else 0)
        M.MODEL_LOADED.set(1)
        M.REF_BRIGHT.set(new.reference.get("brightness_mean") or 0)
        M.REF_CONF.set(new.reference.get("confidence_mean") or 0)
        M.RELOADS.labels("ok").inc()
        log.info("Loaded model version %s (%d classes)", new.version, len(new.classes))
        return True

    def current(self):
        with self._lock:
            return self.model

    def poll_forever(self, interval: float, stop: threading.Event) -> None:
        while not stop.wait(interval):
            self.load()
