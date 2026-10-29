"""Confidence calibration with temperature scaling.

A network trained to near-zero loss on a small dataset becomes overconfident:
it says 99% even when it is wrong. Temperature scaling divides the logits by a
single number T (> 1 softens, < 1 sharpens), chosen to minimise the negative
log-likelihood on the validation set. It never changes which class wins, only
how sure the model claims to be. ECE (expected calibration error) measures the
gap between claimed confidence and actual accuracy: 0 is perfectly honest.
"""
import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, labels, temperature: float = 1.0) -> float:
    p = _softmax(np.asarray(logits, dtype=float) / temperature)
    y = np.asarray(labels, dtype=int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimising validation NLL (log-spaced grid, then a finer grid)."""
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 1.0
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    best = grid[int(np.argmin([nll(logits, labels, t) for t in grid]))]
    fine = np.linspace(best / 1.1, best * 1.1, 100)
    return float(fine[int(np.argmin([nll(logits, labels, t) for t in fine]))])


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| averaged over confidence bins."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 0.0
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def calibrated_probs(logits, temperature: float) -> np.ndarray:
    return _softmax(np.asarray(logits, dtype=float) / temperature)
