"""Shared test fixtures. Run from the repo root:  pytest"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "deploy" / "api")]

CLASSES = ["at_desk", "empty", "in_room", "lying_in_bed", "sitting_on_bed"]


def make_tiny_onnx(path: Path, n_classes: int = len(CLASSES)) -> None:
    """A tiny stand-in for MobileNet with the same input/output contract:
    input (N,3,224,224) -> GlobalAveragePool -> Flatten -> Gemm -> logits (N,K)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(size=(3, n_classes)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(n_classes, dtype=np.float32), "B")
    nodes = [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
             helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
             helper.make_node("Gemm", ["flat", "W", "B"], ["logits"])]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", n_classes])],
        initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def reference(bright_values):
    from roomsense import drift
    ref = drift.reference_stats(bright_values)
    ref["confidence_mean"] = 0.8
    return ref


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("model")
    make_tiny_onnx(d / "model.onnx")
    meta = {"classes": CLASSES, "img_size": 224, "metrics": {"test_macro_f1": 0.5},
            "reference": reference(np.random.default_rng(1).normal(120, 15, 500)), "data_include": {}}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="session")
def client(model_dir, tmp_path_factory):
    """The real FastAPI app, serving the tiny model from a local folder (no MLflow, no camera)."""
    os.environ.update(MODEL_URI=str(model_dir), CAMERA_ENABLED="false", MODEL_POLL_SECONDS="0",
                      STATE_DIR=str(tmp_path_factory.mktemp("state")), DRIFT_MIN_SAMPLES="5",
                      CAPTURE_STATUS_URL="", RETRAIN_TOKEN="")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def jpeg(brightness: int = 120, size=(480, 640)) -> bytes:
    import cv2
    img = np.full((*size, 3), brightness, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()
