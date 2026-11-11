"""The pipeline runner: DAG parsing and reading what `dvc repro` did (with a fake dvc)."""
import os
import stat
from pathlib import Path

from scripts import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["prepare", "preprocess", "split", "train", "export", "register"]


def test_dag_matches_dvc_yaml():
    dag = run_pipeline.parse_dag(str(ROOT / "dvc.yaml"))
    assert [s["name"] for s in dag["stages"]] == ORDER
    edges = {tuple(e) for e in dag["edges"]}
    for a, b in zip(ORDER, ORDER[1:]):
        assert (a, b) in edges                       # the chain
    assert ("split", "export") in edges and ("register", "prepare") not in edges
    assert all(s["cmd"] for s in dag["stages"])


def fake_dvc(tmp_path, monkeypatch, script: str):
    exe = tmp_path / "dvc"
    exe.write_text("#!/bin/sh\n" + script)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def test_successful_run_marks_cached_and_ran_stages(tmp_path, monkeypatch):
    fake_dvc(tmp_path, monkeypatch, """
echo "'data/raw.dvc' didn't change, skipping"
echo "Stage 'prepare' didn't change, skipping"
echo "Running stage 'preprocess':"
echo "> python scripts/preprocess.py"
echo "Running stage 'split':"
exit 0
""")
    r = run_pipeline.run_dvc([], ORDER)
    assert r["returncode"] == 0 and r["failed_stage"] is None
    assert r["stages"]["prepare"]["status"] == "cached"
    assert r["stages"]["preprocess"]["status"] == "ok" and r["stages"]["split"]["status"] == "ok"
    assert "wall_seconds" in r["stages"]["preprocess"]
    assert r["stages"]["train"]["status"] == "not_reached"


def test_failed_run_names_the_failed_stage(tmp_path, monkeypatch):
    fake_dvc(tmp_path, monkeypatch, """
echo "Stage 'prepare' didn't change, skipping"
echo "Running stage 'preprocess':"
echo "Running stage 'train':"
echo "ERROR: failed to reproduce 'train': failed to run: python scripts/train.py, exited with 1"
exit 255
""")
    r = run_pipeline.run_dvc([], ORDER)
    assert r["returncode"] == 255 and r["failed_stage"] == "train"
    assert r["stages"]["train"]["status"] == "failed" and r["stages"]["preprocess"]["status"] == "ok"
    assert r["stages"]["export"]["status"] == "not_reached"
    assert any("failed to reproduce" in line for line in r["log"])


def test_missing_dvc_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))        # no dvc anywhere
    r = run_pipeline.run_dvc([], ORDER)
    assert r["returncode"] == 127 and "could not start dvc" in r["log"][0]
