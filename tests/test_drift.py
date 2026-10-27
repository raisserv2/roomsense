import numpy as np
import pytest

from roomsense import drift


def test_histogram_sums_to_one():
    h = drift.histogram([10, 20, 250, 255])
    assert h.sum() == pytest.approx(1) and len(h) == 16


def test_psi_identical_is_zero_and_shift_is_large():
    ref = drift.histogram(np.random.default_rng(0).normal(120, 15, 2000))
    same = drift.histogram(np.random.default_rng(1).normal(120, 15, 2000))
    shifted = drift.histogram(np.random.default_rng(2).normal(40, 10, 2000))
    assert drift.psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert drift.psi(ref, same) < 0.1
    assert drift.psi(ref, shifted) > 0.25


def test_ood_share_seen_vs_unseen_lighting():
    ref = drift.histogram(np.random.default_rng(0).normal(150, 10, 1000))  # bright room only
    assert drift.ood_share([148, 151, 155], ref) == 0.0      # same lighting
    assert drift.ood_share([30, 35, 40], ref) == 1.0         # lights off: never trained on
    assert drift.ood_share([150, 30], ref) == 0.5


def test_support_mask_spreads_to_neighbours():
    ref = np.zeros(16)
    ref[8] = 1.0
    mask = drift.support_mask(ref, spread=1)
    assert mask[7] and mask[8] and mask[9] and not mask[6] and not mask[10]


def test_window_and_summary():
    w = drift.Window(5)
    for i in range(8):
        w.add(150.0, 0.9, correct=i % 2 == 0)
    assert len(w) == 5
    ref = drift.reference_stats(np.random.default_rng(0).normal(150, 10, 500))
    s = drift.summarise(w, ref, low_conf=0.6)
    assert s["window"] == 5 and s["ood_share"] == 0.0
    assert s["confidence_mean"] == pytest.approx(0.9)
    assert s["low_confidence_ratio"] == 0.0
    assert 0 <= s["live_accuracy"] <= 1


def test_summary_without_reference():
    s = drift.summarise(drift.Window(3), {}, low_conf=0.6)
    assert s["psi"] is None and s["ood_share"] is None and s["live_accuracy"] is None
