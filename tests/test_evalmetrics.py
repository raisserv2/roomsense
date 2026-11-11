"""Macro-F1 must not be dragged down by a class that has no frames in the split."""
import numpy as np

from roomsense.evalmetrics import macro_f1_present, per_class_f1

CLASSES = ["at_desk", "empty", "lights_off"]


def test_perfect_predictions():
    r = macro_f1_present([0, 1, 2, 0], [0, 1, 2, 0], CLASSES)
    assert r["macro_f1"] == 1.0 and r["accuracy"] == 1.0 and r["classes_absent"] == []


def test_absent_class_is_left_out_of_the_average():
    # lights_off (index 2) never occurs in the ground truth
    y_true, y_pred = [0, 0, 1, 1], [0, 0, 1, 1]
    r = macro_f1_present(y_true, y_pred, CLASSES)
    assert r["macro_f1"] == 1.0  # averaging over all three classes would give 0.667
    assert r["classes_absent"] == ["lights_off"] and r["per_class_f1"]["lights_off"] is None
    assert r["classes_evaluated"] == ["at_desk", "empty"]


def test_predicting_an_absent_class_still_costs():
    r = macro_f1_present([0, 0, 1, 1], [0, 2, 1, 1], CLASSES)  # one at_desk frame called lights_off
    assert r["macro_f1"] < 1.0 and r["accuracy"] == 0.75
    assert abs(r["per_class_f1"]["at_desk"] - 2 / 3) < 1e-9


def test_matches_the_textbook_formula():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    f1, support = per_class_f1(y_true, y_pred, 3)
    assert support.tolist() == [3, 2, 1]
    assert np.allclose(f1, [0.8, 0.5, 2 / 3])


def test_empty_input():
    r = macro_f1_present([], [], CLASSES)
    assert r["macro_f1"] == 0.0 and r["n"] == 0 and len(r["classes_absent"]) == 3
