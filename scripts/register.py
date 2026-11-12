"""DVC stage `register`: put the exported model in the MLflow Model Registry and
decide whether it becomes the serving model (alias `champion`).

Promotion gate: the candidate and the current champion are compared on the
candidate's test sessions that the champion never learned from (see
`registry.fair_comparison_set`). The splitter reassigns sessions when the
dataset grows, so part of today's test split may be the champion's old training
data; scoring it there would favour it. The candidate is promoted only if its
macro-F1 on that fair set is at least the champion's + `register.min_delta`,
and only if the fair set has at least `register.min_compare_frames` frames.
(If the class list changed, the two are not comparable and the candidate is
promoted.) Each registered model stores its splits.csv in its MLflow run
(`pipeline/splits.csv`), which is how the next comparison knows what it saw.
The API on nova polls the alias and hot-reloads the new champion within a minute.

Tracking server: $MLFLOW_TRACKING_URI (nova), else local sqlite:///mlflow.db.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.onnx
import numpy as np
import onnx
import pandas as pd
import yaml
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense import registry  # noqa: E402
from roomsense.model_pkg import load_package  # noqa: E402
from roomsense.onnx_eval import macro_f1, session  # noqa: E402
from roomsense.runlog import start_stage  # noqa: E402

stage = start_stage("register")
log = stage.log

PARAMS = yaml.safe_load(open("params.yaml"))
P = PARAMS["register"]
NAME, ALIAS = P["model_name"], P["alias"]
REPORT = Path("reports/register_report.json")

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("roomsense")
client = MlflowClient()

meta = json.loads(Path("models/model_meta.json").read_text())
classes, size = meta["classes"], meta["img_size"]
df = pd.read_csv("data/processed/splits.csv")
test = df[df.split == "test"]

# Re-use the training run when it lives on this tracking server (one run = whole story)
run_id = None
tr = Path("models/train_run.json")
if tr.exists():
    rid = json.loads(tr.read_text()).get("run_id")
    try:
        client.get_run(rid)
        run_id = rid
    except Exception:  # noqa: BLE001  (run was logged to a different server)
        run_id = None

with mlflow.start_run(run_id=run_id, run_name=None if run_id else "register") as run:
    info = mlflow.onnx.log_model(
        onnx.load("models/model.onnx"), name="model", registered_model_name=NAME,
        input_example=np.zeros((1, 3, size, size), dtype=np.float32),
        metadata=meta, save_as_external_data=False)
    mlflow.log_dict(meta, "model_meta.json")
    if Path("reports/export_report.json").exists():
        rep = json.loads(Path("reports/export_report.json").read_text())
        mlflow.log_metrics({f"export_{k}": v for k, v in rep.items() if isinstance(v, (int, float))})
    version = str(info.registered_model_version)

    cand_sess = session("models/model.onnx")
    cand = macro_f1(cand_sess, test, classes, size)

    # Record what this model learned from, for the next comparison against it.
    mlflow.log_artifact("data/processed/splits.csv", artifact_path="pipeline")

    champ_version, champ, reason, compare = None, None, "", None
    try:
        cv = client.get_model_version_by_alias(NAME, ALIAS)
        champ_version = cv.version
    except Exception:  # noqa: BLE001  (no champion yet)
        cv = None

    if cv is None:
        promote, reason = True, "no champion yet"
    elif P.get("force"):
        promote, reason = True, "register.force is true"
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(f"models:/{NAME}/{cv.version}", dst_path=tmp)
            c_onnx, c_meta = load_package(path)
            if list(c_meta["classes"]) != list(classes):
                promote, reason = True, "class list changed"
            else:
                rows, compare = test, None
                try:
                    c_splits = pd.read_csv(mlflow.artifacts.download_artifacts(
                        run_id=cv.run_id, artifact_path="pipeline/splits.csv", dst_path=tmp))
                    rows, compare = registry.fair_comparison_set(test, registry.seen_sessions(c_splits))
                    log.info("comparing on %d frames (%d sessions) the champion never learned from; "
                             "%d test frames excluded", compare["compare_frames"], compare["compare_sessions"],
                             compare["excluded_frames"])
                except Exception as e:  # noqa: BLE001  (champion registered before splits were recorded)
                    log.warning("champion v%s has no recorded split (%s): comparing on the whole test split, "
                                "which may favour it. Backfill with scripts/backfill_splits.py", cv.version, e)
                min_frames = int(P.get("min_compare_frames", 50))
                if compare is not None and compare["compare_frames"] < min_frames:
                    promote = False
                    reason = (f"only {compare['compare_frames']} test frames are unseen by champion v{cv.version} "
                              f"(need {min_frames}); record more sessions or set register.force")
                else:
                    cand_cmp = macro_f1(cand_sess, rows, classes, size)
                    champ = macro_f1(session(c_onnx), rows, classes, size)
                    promote = cand_cmp["macro_f1"] >= champ["macro_f1"] + P["min_delta"]
                    where = (f"{compare['compare_frames']} test frames unseen by both" if compare
                             else "the whole test split (champion's split unknown)")
                    reason = (f"candidate {cand_cmp['macro_f1']:.4f} vs champion v{cv.version} "
                              f"{champ['macro_f1']:.4f} on {where}")
                    if compare:
                        compare.update(candidate_macro_f1=cand_cmp["macro_f1"], champion_macro_f1=champ["macro_f1"])

    client.set_model_version_tag(NAME, version, "test_macro_f1", f"{cand['macro_f1']:.4f}")
    client.set_model_version_tag(NAME, version, "decision", ("promoted: " if promote else "kept champion: ") + reason)
    client.set_model_version_tag(NAME, version, "data_include", json.dumps(meta.get("data_include") or {}))
    client.set_model_version_tag(NAME, version, "git_commit", meta.get("git_commit") or "")
    client.set_model_version_tag(NAME, version, "data_version", meta.get("data_version") or "")
    if promote:
        # Moving the alias is the deployment. The replaced version is remembered for rollback.
        registry.promote(client, NAME, version, ALIAS, reason)
    for extra in sorted(Path("reports").glob("*.json")):
        mlflow.log_artifact(str(extra), artifact_path="pipeline")
    mlflow.log_metrics({"register_candidate_macro_f1": cand["macro_f1"],
                        **({"register_champion_macro_f1": champ["macro_f1"]} if champ else {})})

report = {
    "tracking_uri": mlflow.get_tracking_uri(),
    "run_id": run.info.run_id,
    "model": NAME,
    "version": version,
    "promoted": promote,
    "reason": reason,
    "candidate_test_macro_f1": cand["macro_f1"],
    "champion_version_before": champ_version,
    "champion_compare_macro_f1": champ["macro_f1"] if champ else None,
    "comparison": compare,
    "serving_alias": f"{NAME}@{ALIAS}",
}
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))
log.info("register report: %s", json.dumps(report))
stage.done(items=len(test), version=version, promoted=bool(promote))
