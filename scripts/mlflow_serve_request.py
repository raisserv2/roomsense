"""Send a photo to MLflow's own model server (the optional `mlflow-serve`
compose profile) and print the predicted room state.

    python scripts/mlflow_serve_request.py photo.jpg [http://127.0.0.1:5001]

The image is preprocessed exactly like in training/serving (roomsense.frames),
and the class names come from the champion model's metadata in the registry
(needs MLFLOW_TRACKING_URI; without it, classes are shown as class_0..n).
"""
import json
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense import frames  # noqa: E402

img_path = sys.argv[1]
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5001").rstrip("/")
x = frames.to_input(frames.decode_jpeg(Path(img_path).read_bytes()))
r = requests.post(f"{url}/invocations", json={"inputs": x.tolist()}, timeout=30)
r.raise_for_status()
pred = r.json()["predictions"]
logits = pred["logits"][0] if isinstance(pred, dict) else pred[0]
p = frames.softmax(np.asarray(logits))

classes = None
try:
    import mlflow
    mv = mlflow.MlflowClient().get_model_version_by_alias("roomsense", "champion")
    classes = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/model_meta.json")["classes"]
except Exception:  # noqa: BLE001
    pass
names = classes or [f"class_{i}" for i in range(len(p))]
print(json.dumps({n: round(float(v), 4) for n, v in zip(names, p)}, indent=2))
print("->", names[int(p.argmax())])
