"""Evaluate an ONNX model on a split of the processed dataset (training side only)."""
from pathlib import Path

import cv2
import numpy as np

from roomsense import frames
from roomsense.evalmetrics import macro_f1_present

IMAGES = Path("data/processed/images")


def session(onnx_path, threads: int = 0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def load_bgr(rel: str) -> np.ndarray:
    img = cv2.imread(str(IMAGES / rel))
    if img is None:
        raise FileNotFoundError(IMAGES / rel)
    return img


def predict_paths(sess, paths, size: int = 224, batch: int = 32) -> np.ndarray:
    """Softmax probabilities for a list of processed image paths."""
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(paths), batch):
        x = np.concatenate([frames.to_input(load_bgr(p), size) for p in paths[i:i + batch]])
        out.append(frames.softmax(sess.run(None, {name: x})[0]))
    return np.concatenate(out) if out else np.zeros((0, 0))


def macro_f1(sess, df, classes, size: int = 224) -> dict:
    """Macro-F1 (over the classes present in `df`) and accuracy of an ONNX model on a split."""
    known = df[df.label.isin(classes)]
    probs = predict_paths(sess, known.path.tolist(), size)
    idx = {c: i for i, c in enumerate(classes)}
    y = known.label.map(idx).to_numpy()
    p = probs.argmax(1) if len(probs) else np.array([], dtype=int)
    out = macro_f1_present(y, p, classes)
    out["confidence_mean"] = float(probs.max(1).mean()) if len(probs) else None
    return out
