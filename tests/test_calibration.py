import numpy as np
import pytest

from roomsense import calibration


def overconfident(n=4000, k=5, seed=0):
    """Logits from a model that is right ~70% of the time but claims ~99%."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), pred] += 8.0
    return logits, y


def test_temperature_softens_an_overconfident_model():
    logits, y = overconfident()
    t = calibration.fit_temperature(logits, y)
    assert t > 1.5
    before = calibration.ece(calibration.calibrated_probs(logits, 1.0), y)
    after = calibration.ece(calibration.calibrated_probs(logits, t), y)
    assert after < before / 2


def test_temperature_never_changes_the_prediction():
    logits, _ = overconfident(200)
    assert (calibration.calibrated_probs(logits, 3.0).argmax(1) == logits.argmax(1)).all()


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert calibration.ece(probs, [0, 1]) == pytest.approx(0)
    assert calibration.ece(probs, [1, 0]) == pytest.approx(1)


def test_empty_inputs():
    assert calibration.fit_temperature(np.zeros((0, 3)), []) == 1.0
    assert calibration.ece(np.zeros((0, 3)), []) == 0.0
