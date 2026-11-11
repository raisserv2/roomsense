"""DVC stage `split`: assign whole sessions to train / val / test.

Frames of one recording session look almost identical, so a session must never
appear in two splits (that would leak test data into training). Sessions are
assigned so that every class keeps roughly the requested share in each split:
random restarts plus greedy single-session moves minimise the squared distance
to the per-class target, with a heavy penalty for a class that is absent from
val or test.

`data.include` in params.yaml optionally filters the frames first, for example
{lighting: [tube_on]}. That is how the drift demo trains a model that has only
ever seen the tube light on.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import start_stage  # noqa: E402

NAMES = ["train", "val", "test"]


def load_include(path: str = "params.yaml") -> dict:
    """The optional data filter from params.yaml ({} = keep everything)."""
    if not Path(path).exists():
        return {}
    return ((yaml.safe_load(Path(path).read_text()) or {}).get("data") or {}).get("include") or {}


def apply_include(rows: list, include: dict) -> list:
    """Keep only the rows whose column value is in the allowed list, for every filtered column."""
    for col, allowed in include.items():
        allowed = {str(v) for v in (allowed if isinstance(allowed, list) else [allowed])}
        rows = [r for r in rows if r.get(col) in allowed]
    return rows


def assign_sessions(rows: list, fracs: dict, seed: int, restarts: int = 200) -> tuple:
    """Return ({session: split}, {split: Counter(label -> frames)})."""
    by_sess = defaultdict(Counter)
    for r in rows:
        by_sess[r["session_id"]][r["label"]] += 1
    total = Counter(r["label"] for r in rows)
    classes = list(total)
    target = {s: {c: total[c] * fr for c in classes} for s, fr in fracs.items()}
    sessions = list(by_sess)
    rng = random.Random(seed)

    def cost(assign):
        have = {s: Counter() for s in NAMES}
        for sess, sp in assign.items():
            have[sp].update(by_sess[sess])
        c = 0.0
        for s in NAMES:
            for cl in classes:
                c += ((have[s][cl] - target[s][cl]) / total[cl]) ** 2
                if s != "train" and have[s][cl] == 0:
                    c += 1.0  # heavy penalty: a class absent from val/test
        return c, have

    best = None
    for _ in range(restarts):  # random restarts + greedy session moves
        assign = {s: rng.choices(NAMES, [fracs[n] for n in NAMES])[0] for s in sessions}
        cur, _h = cost(assign)
        improved = True
        while improved:
            improved = False
            for s in sessions:
                for sp in NAMES:
                    if sp == assign[s]:
                        continue
                    old = assign[s]
                    assign[s] = sp
                    c, _h = cost(assign)
                    if c < cur - 1e-12:
                        cur, improved = c, True
                    else:
                        assign[s] = old
        if best is None or cur < best[0]:
            best = (cur, dict(assign))
    assign = best[1]
    return assign, cost(assign)[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    a = p.parse_args()
    stage = start_stage("split")

    with open("data/processed/clean_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    include = load_include()
    n_before = len(rows)
    rows = apply_include(rows, include)
    if not rows:
        raise SystemExit(f"data.include {include} removed every frame")

    fracs = {"train": 1 - a.val - a.test, "val": a.val, "test": a.test}
    assign, have = assign_sessions(rows, fracs, a.seed)
    classes = list(Counter(r["label"] for r in rows))
    for r in rows:
        r["split"] = assign[r["session_id"]]
    with open("data/processed/splits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "seed": a.seed,
        "include": include,
        "frames_before_filter": n_before,
        "sessions": dict(Counter(assign.values())),
        "frames": {s: sum(have[s].values()) for s in NAMES},
        "frame_share": {s: round(sum(have[s].values()) / len(rows), 3) for s in NAMES},
        "per_class": {s: dict(have[s]) for s in NAMES},
        "classes_missing_from_val_or_test": [
            f"{s}:{c}" for s in ("val", "test") for c in classes if have[s][c] == 0],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/split_report.json").write_text(json.dumps(report, indent=2))
    if report["classes_missing_from_val_or_test"]:
        stage.log.warning("classes missing from val/test (too few sessions): %s",
                          report["classes_missing_from_val_or_test"])
    stage.log.info("split: %s", report["frames"])
    stage.done(items=len(rows))


if __name__ == "__main__":
    main()
