import argparse, csv, json, random
from collections import Counter, defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--val", type=float, default=0.15)
p.add_argument("--test", type=float, default=0.15)
a = p.parse_args()

with open("data/processed/manifest.csv", newline="") as f:
    rows = list(csv.DictReader(f))

by_sess = defaultdict(Counter)
for r in rows:
    by_sess[r["session_id"]][r["label"]] += 1
total = Counter(r["label"] for r in rows)
classes = list(total)

fracs = {"train": 1 - a.val - a.test, "val": a.val, "test": a.test}
names = list(fracs)
target = {s: {c: total[c] * fr for c in classes} for s, fr in fracs.items()}
sessions = list(by_sess)
rng = random.Random(a.seed)

def cost(assign):
    have = {s: Counter() for s in names}
    for sess, sp in assign.items():
        have[sp].update(by_sess[sess])
    c = 0.0
    for s in names:
        for cl in classes:
            c += ((have[s][cl] - target[s][cl]) / total[cl]) ** 2
            if s != "train" and have[s][cl] == 0:
                c += 1.0  # heavy penalty: a class absent from val/test
    return c, have

best = None
for _ in range(200):  # random restarts + greedy session moves
    assign = {s: rng.choices(names, [fracs[n] for n in names])[0] for s in sessions}
    cur, _h = cost(assign)
    improved = True
    while improved:
        improved = False
        for s in sessions:
            for sp in names:
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
_c, have = cost(assign)
for r in rows:
    r["split"] = assign[r["session_id"]]
with open("data/processed/splits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

report = {
    "seed": a.seed,
    "sessions": dict(Counter(assign.values())),
    "frames": {s: sum(have[s].values()) for s in names},
    "frame_share": {s: round(sum(have[s].values()) / len(rows), 3) for s in names},
    "per_class": {s: dict(have[s]) for s in names},
    "classes_missing_from_val_or_test": [
        f"{s}:{c}" for s in ("val", "test") for c in classes if have[s][c] == 0],
}
Path("reports").mkdir(exist_ok=True)
with open("reports/split_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
