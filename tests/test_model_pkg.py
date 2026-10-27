import json

import pytest
import yaml

from roomsense.model_pkg import load_package
from tests.conftest import CLASSES, make_tiny_onnx


def test_plain_export_folder(model_dir):
    path, meta = load_package(model_dir)
    assert path.name == "model.onnx" and meta["classes"] == CLASSES


def test_mlflow_model_folder(tmp_path):
    make_tiny_onnx(tmp_path / "model.onnx")
    (tmp_path / "MLmodel").write_text(yaml.safe_dump({"flavors": {}, "metadata": {"classes": ["a", "b"]}}))
    _, meta = load_package(tmp_path)
    assert meta["classes"] == ["a", "b"]


def test_missing_classes_is_an_error(tmp_path):
    make_tiny_onnx(tmp_path / "model.onnx")
    (tmp_path / "model_meta.json").write_text(json.dumps({"img_size": 224}))
    with pytest.raises(ValueError):
        load_package(tmp_path)


def test_missing_onnx_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_package(tmp_path)
