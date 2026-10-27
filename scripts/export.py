"""DVC stage `export`: PyTorch checkpoint -> ONNX model package for CPU serving.

1. Rebuild MobileNetV3-Small, load models/best.pt, export models/model.onnx.
2. Parity check: ONNX Runtime output must match PyTorch on real validation frames.
3. CPU latency benchmark (what nova's i5 will see).
4. Evaluate the ONNX model on the test split.
5. Reference statistics from the training frames (brightness histogram) and the
   validation confidence: the baseline the live drift monitor compares against.
6. Write models/model_meta.json and reports/export_report.json.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense import drift, frames  # noqa: E402
from roomsense.onnx_eval import load_bgr, macro_f1, session  # noqa: E402

PARAMS = yaml.safe_load(open("params.yaml"))
P = PARAMS["export"]
SIZE = PARAMS["train"]["img_h"]
OUT, META, REPORT = Path("models/model.onnx"), Path("models/model_meta.json"), Path("reports/export_report.json")

classes = json.load(open("models/classes.json"))
df = pd.read_csv("data/processed/splits.csv")

# 1. rebuild + export
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load("models/best.pt", map_location="cpu"))
model.eval()
dummy = torch.zeros(1, 3, SIZE, SIZE)
kw = dict(input_names=["input"], output_names=["logits"], opset_version=P["opset"],
          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
exporter = "torchscript"
try:
    torch.onnx.export(model, dummy, str(OUT), dynamo=False, **kw)
except Exception as e:  # noqa: BLE001  (old exporter removed/broken: use the dynamo one)
    print(f"TorchScript exporter failed ({e}); falling back to the dynamo exporter")
    exporter = "dynamo"
    torch.onnx.export(model, dummy, str(OUT), dynamo=True, external_data=False,
                      input_names=["input"], output_names=["logits"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}})
import onnx  # noqa: E402
onnx.checker.check_model(onnx.load(str(OUT)))

# 2. parity on real frames
val = df[df.split == "val"].head(P["parity_frames"])
sess = session(OUT)
x = np.concatenate([frames.to_input(load_bgr(p), SIZE) for p in val.path]) if len(val) else dummy.numpy()
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
max_diff = float(np.abs(ref - got).max())
agree = float((ref.argmax(1) == got.argmax(1)).mean())
assert max_diff < P["parity_tol"], f"ONNX/PyTorch mismatch: max |diff| = {max_diff}"

# 3. CPU latency, single frame, few threads (like nova)
bench = session(OUT, threads=P["bench_threads"])
one = x[:1]
for _ in range(5):
    bench.run(None, {"input": one})
t = []
for _ in range(P["bench_runs"]):
    t0 = time.perf_counter()
    bench.run(None, {"input": one})
    t.append((time.perf_counter() - t0) * 1000)
frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
tp = []
for _ in range(20):
    t0 = time.perf_counter()
    frames.to_input(frame, SIZE)
    tp.append((time.perf_counter() - t0) * 1000)

# 4. test metrics of the exported model
test = macro_f1(sess, df[df.split == "test"], classes, SIZE)
valm = macro_f1(sess, df[df.split == "val"], classes, SIZE)

# 5. reference statistics for drift monitoring
train_paths = df[df.split == "train"].path.tolist()
bright = [frames.brightness(load_bgr(p), SIZE) for p in train_paths]
reference = drift.reference_stats(bright)
reference["confidence_mean"] = valm["confidence_mean"]

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    commit = ""
train_metrics = json.load(open("reports/metrics.json")) if Path("reports/metrics.json").exists() else {}
meta = {
    "name": "roomsense",
    "classes": classes,
    "img_size": SIZE,
    "input_name": "input",
    "output_name": "logits",
    "normalisation": {"mean": frames.MEAN.tolist(), "std": frames.STD.tolist()},
    "reference": reference,
    "metrics": {"test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
                "val_macro_f1": valm["macro_f1"],
                "torch_test_macro_f1": train_metrics.get("test_macro_f1")},
    "data_include": (PARAMS.get("data") or {}).get("include") or {},
    "train_frames": len(train_paths),
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": commit,
}
META.write_text(json.dumps(meta, indent=2))
report = {
    "exporter": exporter,
    "opset": P["opset"],
    "onnx_size_mb": round(OUT.stat().st_size / 1e6, 2),
    "parity_max_abs_diff": max_diff,
    "parity_argmax_agreement": agree,
    "latency_ms_p50": float(np.percentile(t, 50)),
    "latency_ms_p95": float(np.percentile(t, 95)),
    "preprocess_ms_p50_720p": float(np.percentile(tp, 50)),
    "bench_threads": P["bench_threads"],
    "onnx_test_macro_f1": test["macro_f1"],
    "onnx_test_accuracy": test["accuracy"],
    "reference_brightness_mean": reference["brightness_mean"],
    "reference_confidence_mean": reference["confidence_mean"],
}
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
