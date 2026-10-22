import csv
from datetime import datetime
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

rows, missing = [], 0
with open(RAW / "frames.csv", newline="") as f:
    for r in csv.DictReader(f):
        if not (RAW / r["path"]).is_file():
            missing += 1
            continue
        hour = datetime.fromisoformat(r["timestamp"]).hour
        r["period"] = "day" if 6 <= hour < 18 else "night"
        rows.append(r)

fields = ["path", "session_id", "label", "lighting", "timestamp", "period"]
with open(OUT / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} frames kept, {missing} dropped (file missing)")
