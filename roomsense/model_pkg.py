"""Read a RoomSense model package: an ONNX file plus its metadata.

A package is either an MLflow ONNX model directory (MLmodel + model.onnx, with
our metadata stored under the MLmodel `metadata` key) or a plain directory with
model.onnx + model_meta.json (what the export stage writes).
"""
import json
from pathlib import Path

import yaml


def load_package(path) -> tuple:
    """Return (onnx_path, meta) for a package directory."""
    p = Path(path)
    if p.is_file() and p.suffix == ".onnx":
        p = p.parent
    onnx_files = sorted(p.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file under {p}")
    meta = {}
    if (p / "MLmodel").exists():
        meta = (yaml.safe_load((p / "MLmodel").read_text()) or {}).get("metadata") or {}
    if not meta and (p / "model_meta.json").exists():
        meta = json.loads((p / "model_meta.json").read_text())
    if "classes" not in meta:
        raise ValueError(f"Model package {p} has no class list in its metadata")
    return onnx_files[0], meta
