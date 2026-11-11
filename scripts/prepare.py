"""DVC stage `prepare`: data ingestion check. raw frames.csv -> a validated manifest.

Reads the frame list written by the capture service (data/raw/frames.csv),
runs the data quality checks below on every row, adds the day/night period,
and writes data/processed/manifest.csv plus reports/prepare_report.json.

Quality checks (a failing row is dropped and counted by reason):
  * the image file exists and is not empty
  * the label and the lighting tag are known values, and the pair is possible
  * the timestamp parses
  * the same file is not listed twice
The stage fails when more than `prepare.max_bad_share` of the rows are dropped,
so a broken snapshot can never silently train a model.
"""
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import start_stage  # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/processed")
REPORT = Path("reports/prepare_report.json")
FIELDS = ["path", "session_id", "label", "lighting", "timestamp", "period"]
REQUIRED = FIELDS[:-1]
DEFAULTS = {
    "labels": ["at_desk", "empty", "in_room", "lights_off", "lying_in_bed", "sitting_on_bed"],
    "lighting": ["tube_on", "tube_off"],
    "invalid_combos": [["lights_off", "tube_on"]],
    "max_bad_share": 0.05,
}


def load_params(path: str = "params.yaml") -> dict:
    """The `prepare` section of params.yaml on top of the defaults."""
    params = dict(DEFAULTS)
    if Path(path).exists():
        params.update((yaml.safe_load(Path(path).read_text()) or {}).get("prepare") or {})
    return params


def check_row(row: dict, params: dict, seen: set, raw: Path = RAW):
    """Return None for a good row, else the reason it is dropped."""
    if any(not row.get(k) for k in REQUIRED):
        return "missing_field"
    if row["path"] in seen:
        return "duplicate_path"
    if row["label"] not in params["labels"]:
        return "unknown_label"
    if row["lighting"] not in params["lighting"]:
        return "unknown_lighting"
    if [row["label"], row["lighting"]] in [list(c) for c in params["invalid_combos"]]:
        return "impossible_combo"
    try:
        datetime.fromisoformat(row["timestamp"])
    except ValueError:
        return "bad_timestamp"
    file = raw / row["path"]
    if not file.is_file():
        return "file_missing"
    if file.stat().st_size == 0:
        return "file_empty"
    return None


def period_of(timestamp: str) -> str:
    """06:00-17:59 counts as day, the rest as night."""
    return "day" if 6 <= datetime.fromisoformat(timestamp).hour < 18 else "night"


def prepare(params: dict, raw: Path = RAW) -> tuple:
    """(kept rows, report dict). Raises ValueError when the data is not usable."""
    manifest = raw / "frames.csv"
    if not manifest.is_file():
        raise ValueError(f"{manifest} not found: run `dvc pull` first")
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"frames.csv lacks the columns {missing_cols}")
        rows = list(reader)

    kept, dropped, seen = [], Counter(), set()
    for row in rows:
        reason = check_row(row, params, seen, raw)
        if reason:
            dropped[reason] += 1
            continue
        seen.add(row["path"])
        row["period"] = period_of(row["timestamp"])
        kept.append({k: row[k] for k in FIELDS})

    bad_share = (len(rows) - len(kept)) / len(rows) if rows else 1.0
    sessions = Counter(r["session_id"] for r in kept)
    report = {
        "frames_in": len(rows),
        "frames_kept": len(kept),
        "dropped": dict(dropped),
        "bad_share": round(bad_share, 4),
        "max_bad_share": params["max_bad_share"],
        "sessions": len(sessions),
        "per_class": dict(Counter(r["label"] for r in kept)),
        "per_lighting": dict(Counter(r["lighting"] for r in kept)),
        "per_period": dict(Counter(r["period"] for r in kept)),
        "sessions_per_class": dict(Counter(lab for lab, _sid in {(r["label"], r["session_id"]) for r in kept})),
        "first_frame": min((r["timestamp"] for r in kept), default=None),
        "last_frame": max((r["timestamp"] for r in kept), default=None),
        "classes_without_frames": [c for c in params["labels"] if not any(r["label"] == c for r in kept)],
    }
    if not kept:
        raise ValueError("No usable frames in data/raw")
    if bad_share > params["max_bad_share"]:
        raise ValueError(f"{bad_share:.1%} of the rows failed the quality checks "
                         f"(limit {params['max_bad_share']:.0%}): {dict(dropped)}")
    return kept, report


def main() -> None:
    stage = start_stage("prepare")
    kept, report = prepare(load_params())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    stage.log.info("%d frames kept, %d dropped %s", len(kept), report["frames_in"] - len(kept), report["dropped"])
    stage.done(items=report["frames_in"])


if __name__ == "__main__":
    main()
