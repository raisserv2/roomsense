"""Stage timing records and provenance (reports/stages/*.json)."""
import json
import logging

from roomsense import runlog


def test_stage_record_has_duration_and_throughput(tmp_path):
    run = runlog.StageRun("demo", logging.getLogger("t"), out_dir=tmp_path)
    assert json.loads((tmp_path / "demo.json").read_text())["status"] == "running"
    record = run.done(items=500, unit="frames", extra_note="x")
    saved = json.loads((tmp_path / "demo.json").read_text())
    assert saved == record and saved["status"] == "ok"
    assert saved["items"] == 500 and saved["throughput_per_s"] > 0 and saved["duration_s"] >= 0
    assert saved["extra_note"] == "x" and saved["started_at"] <= saved["finished_at"]


def test_failed_stage_keeps_the_error(tmp_path):
    run = runlog.StageRun("demo", logging.getLogger("t"), out_dir=tmp_path)
    run.fail("ValueError: no frames")
    saved = runlog.read_stage("demo", tmp_path)
    assert saved["status"] == "failed" and "no frames" in saved["error"]


def test_read_stage_of_a_stage_that_never_ran(tmp_path):
    assert runlog.read_stage("nope", tmp_path) == {}


def test_data_version_reads_the_dvc_pointer(tmp_path):
    f = tmp_path / "raw.dvc"
    f.write_text("outs:\n- md5: 0123abcd.dir\n  size: 10\n  path: raw\n")
    assert runlog.data_version(str(f)) == "0123abcd.dir"
    assert runlog.data_version(str(tmp_path / "missing.dvc")) == ""


def test_provenance_keys():
    assert set(runlog.provenance()) == {"git_commit", "git_dirty", "data_version"}
