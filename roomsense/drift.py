"""Drift statistics on input brightness (mean grey level of a frame).

Two signals compare a window of recent live frames with the training frames:

* Out-of-distribution share (the alerting signal): the share of live frames
  whose brightness falls in a range the model (almost) never saw in training.
  A 10-minute window naturally sits in one lighting condition, so it should be
  judged by "is this condition covered by training?", not by "does this short
  window have the same spread as months of training data?".

* PSI, Population Stability Index (informational):
      PSI = sum((cur - ref) * ln(cur / ref))  over brightness bins.
  Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large shift. On a
  short window PSI is high even without drift, so it is shown, not alerted on.
"""
from collections import deque
from threading import Lock

import numpy as np

BIN_EDGES = np.linspace(0, 256, 17)  # 16 equal brightness bins over 0-255
EPS = 1e-4


def histogram(values, edges=BIN_EDGES) -> np.ndarray:
    """Share of values per bin (sums to 1). Empty input -> uniform-free zeros."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), bins=edges)
    return counts / counts.sum()


def psi(reference, current, eps: float = EPS) -> float:
    """Population Stability Index between two bin-share vectors."""
    r = np.clip(np.asarray(reference, dtype=float), eps, None)
    c = np.clip(np.asarray(current, dtype=float), eps, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def support_mask(reference_hist, min_share: float = 0.005, spread: int = 1) -> np.ndarray:
    """Bins covered by training data: share >= min_share, widened by `spread` bins."""
    seen = np.asarray(reference_hist, dtype=float) >= min_share
    mask = seen.copy()
    for k in range(1, spread + 1):
        mask[k:] |= seen[:-k]
        mask[:-k] |= seen[k:]
    return mask


def ood_share(values, reference_hist, min_share: float = 0.005, spread: int = 1, edges=BIN_EDGES) -> float:
    """Share of values falling in brightness bins the training data does not cover."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return 0.0
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
    return float(np.mean(~support_mask(reference_hist, min_share, spread)[idx]))


def reference_stats(brightness_values, confidences=None) -> dict:
    """Summary of the training data that live traffic is compared against."""
    b = np.asarray(list(brightness_values), dtype=float)
    ref = {
        "brightness_edges": BIN_EDGES.tolist(),
        "brightness_hist": histogram(b).tolist(),
        "brightness_mean": float(b.mean()) if b.size else None,
        "brightness_std": float(b.std()) if b.size else None,
        "n": int(b.size),
    }
    if confidences is not None and len(confidences):
        ref["confidence_mean"] = float(np.mean(confidences))
    return ref


class Window:
    """Thread-safe rolling window of recent live observations."""

    def __init__(self, size: int):
        self.size = size
        self._b = deque(maxlen=size)      # brightness
        self._c = deque(maxlen=size)      # top-class confidence
        self._ok = deque(maxlen=size)     # 1/0 when ground truth was known
        self._lock = Lock()

    def add(self, brightness: float, confidence: float, correct=None) -> None:
        with self._lock:
            self._b.append(brightness)
            self._c.append(confidence)
            if correct is not None:
                self._ok.append(1.0 if correct else 0.0)

    def snapshot(self) -> dict:
        with self._lock:
            return {"brightness": list(self._b), "confidence": list(self._c), "correct": list(self._ok)}

    def __len__(self) -> int:
        with self._lock:
            return len(self._b)


def summarise(window: Window, reference: dict, low_conf: float, min_share: float = 0.005) -> dict:
    """Everything the API and the dashboard need about the current window."""
    s = window.snapshot()
    b, c, ok = s["brightness"], s["confidence"], s["correct"]
    cur = histogram(b)
    out = {
        "window": len(b),
        "ood_share": ood_share(b, reference["brightness_hist"], min_share) if b and reference else None,
        "psi": psi(reference["brightness_hist"], cur) if b and reference else None,
        "brightness_mean": float(np.mean(b)) if b else None,
        "reference_brightness_mean": reference.get("brightness_mean") if reference else None,
        "current_hist": cur.tolist(),
        "reference_hist": reference.get("brightness_hist") if reference else None,
        "confidence_mean": float(np.mean(c)) if c else None,
        "reference_confidence_mean": reference.get("confidence_mean") if reference else None,
        "low_confidence_ratio": float(np.mean(np.asarray(c) < low_conf)) if c else None,
        "live_accuracy": float(np.mean(ok)) if ok else None,
        "labelled": len(ok),
    }
    return out
