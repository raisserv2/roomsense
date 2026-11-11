"""Classification metrics that stay fair when a class is missing from a split.

With few recording sessions a rare class (``lights_off``) can end up with zero
frames in the test split. scikit-learn's macro-F1 over *all* classes then counts
an F1 of 0 for that class and drags the average down for a reason that has
nothing to do with the model. ``macro_f1_present`` averages only over the
classes that actually occur in the ground truth, and says which were skipped.
Predictions of an absent class still count as mistakes for the true class.
"""
import numpy as np


def per_class_f1(y_true, y_pred, n_classes: int) -> tuple:
    """(F1 per class, number of true samples per class)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    f1, support = np.zeros(n_classes), np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support[c] = tp + fn
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1, support


def macro_f1_present(y_true, y_pred, classes) -> dict:
    """Macro-F1 over the classes present in ``y_true``, plus the per-class detail."""
    classes = list(classes)
    f1, support = per_class_f1(y_true, y_pred, len(classes))
    present = support > 0
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    return {
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "per_class_f1": {c: (float(f1[i]) if present[i] else None) for i, c in enumerate(classes)},
        "support": {c: int(support[i]) for i, c in enumerate(classes)},
        "classes_evaluated": [c for i, c in enumerate(classes) if present[i]],
        "classes_absent": [c for i, c in enumerate(classes) if not present[i]],
        "n": int(len(y_true)),
    }
