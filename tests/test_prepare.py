"""Data quality checks of the prepare stage."""
import csv

import pytest

from scripts import prepare

FIELDS = ["path", "session_id", "label", "lighting", "timestamp"]


def make_raw(tmp_path, rows, create=True):
    raw = tmp_path / "raw"
    raw.mkdir()
    with open(raw / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if create:
        for r in rows:
            p = raw / r["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"jpeg")
    return raw


def row(i, label="at_desk", lighting="tube_on", ts="2026-11-01T10:00:00+05:30", session="s1"):
    return {"path": f"{session}/{i}.jpg", "session_id": session, "label": label, "lighting": lighting, "timestamp": ts}


def test_good_rows_pass_and_get_a_period(tmp_path):
    rows = [row(1), row(2, ts="2026-11-01T22:30:00+05:30", session="s2")]
    kept, report = prepare.prepare(dict(prepare.DEFAULTS), make_raw(tmp_path, rows))
    assert [r["period"] for r in kept] == ["day", "night"]
    assert report["frames_in"] == 2 and report["frames_kept"] == 2 and report["dropped"] == {}
    assert report["sessions"] == 2 and "lights_off" in report["classes_without_frames"]


def test_every_kind_of_bad_row_is_dropped_and_counted(tmp_path):
    rows = [row(i) for i in range(40)] + [
        row(100, label="dancing"),                      # unknown label
        row(101, lighting="candle"),                    # unknown lighting
        row(102, label="lights_off"),                   # lights_off with tube_on cannot happen
        row(103, ts="yesterday"),                       # bad timestamp
        row(0),                                         # duplicate path
    ]
    raw = make_raw(tmp_path, rows)
    (raw / "s1/5.jpg").unlink()                         # file missing
    (raw / "s1/6.jpg").write_bytes(b"")                 # empty file
    params = {**prepare.DEFAULTS, "max_bad_share": 0.5}
    kept, report = prepare.prepare(params, raw)
    assert report["dropped"] == {"unknown_label": 1, "unknown_lighting": 1, "impossible_combo": 1,
                                 "bad_timestamp": 1, "duplicate_path": 1, "file_missing": 1, "file_empty": 1}
    assert len(kept) == 38


def test_too_many_bad_rows_fail_the_stage(tmp_path):
    rows = [row(1), row(2, label="dancing"), row(3, label="dancing")]
    with pytest.raises(ValueError, match="failed the quality checks"):
        prepare.prepare(dict(prepare.DEFAULTS), make_raw(tmp_path, rows))


def test_missing_manifest_and_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        prepare.prepare(dict(prepare.DEFAULTS), tmp_path / "nowhere")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "frames.csv").write_text("path,label\na.jpg,at_desk\n")
    with pytest.raises(ValueError, match="lacks the columns"):
        prepare.prepare(dict(prepare.DEFAULTS), raw)
