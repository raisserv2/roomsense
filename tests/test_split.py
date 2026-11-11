"""The split stage must never leak a session across train/val/test."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_split(tmp_path, include=None):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/split.py", tmp_path / "scripts/split.py")
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({"data": {"include": include or {}}}))
    rows = []
    labels = ["at_desk", "empty", "in_room", "lying_in_bed", "sitting_on_bed"]
    for s in range(30):
        for f in range(10):
            rows.append({"path": f"s{s}/{f}.jpg", "session_id": f"s{s}", "label": labels[s % 5],
                         "lighting": "tube_on" if s % 2 else "tube_off", "period": "day"})
    with open(tmp_path / "data/processed/clean_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # the copied script imports the shared `roomsense` package from the repo
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "scripts/split.py", "--seed", "42"], cwd=tmp_path, check=True,
                   capture_output=True, env=env)
    with open(tmp_path / "data/processed/splits.csv", newline="") as fh:
        out = list(csv.DictReader(fh))
    return out, json.loads((tmp_path / "reports/split_report.json").read_text())


def test_no_session_in_two_splits(tmp_path):
    rows, report = run_split(tmp_path)
    where = {}
    for r in rows:
        where.setdefault(r["session_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in where.values())
    assert report["classes_missing_from_val_or_test"] == []
    assert abs(report["frame_share"]["train"] - 0.7) < 0.1


def test_include_filter(tmp_path):
    rows, report = run_split(tmp_path, include={"lighting": ["tube_on"]})
    assert {r["lighting"] for r in rows} == {"tube_on"}
    assert report["frames_before_filter"] == 300 and len(rows) == 150


def test_split_records_its_stage_timing(tmp_path):
    run_split(tmp_path)
    record = json.loads((tmp_path / "reports/stages/split.json").read_text())
    assert record["status"] == "ok" and record["items"] == 300 and record["throughput_per_s"] > 0


def test_split_is_reproducible(tmp_path):
    a, _ = run_split(tmp_path / "a")
    b, _ = run_split(tmp_path / "b")
    assert [r["split"] for r in a] == [r["split"] for r in b]
