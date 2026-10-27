"""Live monitoring: camera loop, ground truth from the capture service, drift window."""
import logging
import threading
import time

import requests

from roomsense import drift, frames

from . import metrics as M
from . import settings as S

log = logging.getLogger("monitor")


class Monitor:
    def __init__(self, models):
        self.models = models
        self.window = drift.Window(S.DRIFT_WINDOW)
        self.latest = None          # last camera prediction (dict)
        self.latest_jpeg = None     # last camera frame bytes
        self.camera_error = None
        self._lock = threading.Lock()

    # --- shared by camera frames and monitored uploads -------------------
    def observe(self, img_bgr, source: str, truth=None, jpeg: bytes = None) -> dict:
        model = self.models.current()
        if model is None:
            raise RuntimeError("No model loaded yet")
        res = model.predict(img_bgr)
        b = frames.brightness(img_bgr, model.size)
        M.LATENCY.labels(source).observe(res.pop("_latency_s"))
        M.PREDICTIONS.labels(res["label"], source).inc()
        M.CONFIDENCE.observe(res["confidence"])
        M.BRIGHTNESS.observe(b)
        correct = None
        if truth and truth in model.classes:
            correct = truth == res["label"]
            M.LABELLED.labels(str(correct).lower()).inc()
        self.window.add(b, res["confidence"], correct)
        res.update(brightness=b, source=source, truth=truth, correct=correct,
                   model_version=model.version, timestamp=time.time())
        if source == "camera":
            for c in model.classes:
                M.CURRENT.labels(c).set(1 if c == res["label"] else 0)
            with self._lock:
                self.latest, self.latest_jpeg = res, jpeg
        self.update_drift_metrics()
        return res

    def drift_summary(self) -> dict:
        model = self.models.current()
        ref = model.reference if model else {}
        out = drift.summarise(self.window, ref, S.LOW_CONFIDENCE, S.OOD_MIN_SHARE)
        out["threshold"] = S.OOD_THRESHOLD
        out["min_samples"] = S.DRIFT_MIN_SAMPLES
        out["drifting"] = bool(out["ood_share"] is not None and out["window"] >= S.DRIFT_MIN_SAMPLES
                               and out["ood_share"] > S.OOD_THRESHOLD)
        return out

    def update_drift_metrics(self) -> None:
        d = self.drift_summary()
        M.WINDOW.set(d["window"])
        if d["psi"] is not None:
            M.PSI.set(d["psi"])
        if d["ood_share"] is not None:
            M.OOD.set(d["ood_share"])
        if d["brightness_mean"] is not None:
            M.BRIGHT_MEAN.set(d["brightness_mean"])
        if d["confidence_mean"] is not None:
            M.CONF_MEAN.set(d["confidence_mean"])
            M.LOW_CONF.set(d["low_confidence_ratio"])
        M.LIVE_ACC.set(d["live_accuracy"] if d["live_accuracy"] is not None else -1)

    # --- camera loop -------------------------------------------------------
    def _truth(self):
        if not S.CAPTURE_STATUS_URL:
            return None
        try:
            s = requests.get(S.CAPTURE_STATUS_URL, timeout=2).json()
            return (s.get("session") or {}).get("label")
        except (requests.RequestException, ValueError):
            return None

    def camera_once(self) -> None:
        try:
            auth = (S.CAM_USER, S.CAM_PASS) if S.CAM_USER else None
            r = requests.get(S.CAM_URL, auth=auth, timeout=5)
            r.raise_for_status()
            img = frames.decode_jpeg(r.content)
        except (requests.RequestException, ValueError) as e:
            self.camera_error = str(e)
            M.CAMERA_ERRORS.inc()
            return
        self.camera_error = None
        M.LAST_FRAME.set(time.time())
        if self.models.current() is None:
            with self._lock:
                self.latest_jpeg = r.content
            return
        self.observe(img, "camera", truth=self._truth(), jpeg=r.content)

    def camera_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            t0 = time.time()
            try:
                self.camera_once()
            except Exception:  # noqa: BLE001  (never let the loop die)
                log.exception("camera loop error")
            stop.wait(max(0.5, S.CAMERA_INTERVAL - (time.time() - t0)))
