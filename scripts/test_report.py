"""Run the tests, check the acceptance criteria, write the test report.

    python scripts/test_report.py                                   # tests + lint + criteria from reports/
    python scripts/test_report.py --api https://nova.tailb5cb87.ts.net:4443/api   # plus live checks on nova

Outputs:
  reports/tests/junit.xml     raw pytest results (JUnit XML)
  reports/acceptance.json     every acceptance criterion with its measured value
  docs/05-test-report.md      the human-readable report (counts, every test case, criteria)

The acceptance thresholds live in params.yaml (`acceptance:`), so they are
versioned with the code. Exit code 0 only when no test failed and no criterion
that could be measured failed.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense.runlog import git_commit, setup_logging  # noqa: E402

log = setup_logging("test-report")
JUNIT = Path("reports/tests/junit.xml")
ACCEPT = Path("reports/acceptance.json")
REPORT = Path("docs/05-test-report.md")
DEFAULTS = {"macro_f1_min": 0.60, "latency_ms_p95_max": 100.0, "api_latency_ms_p95_max": 500.0,
            "parity_max_abs_diff": 0.001, "data_bad_share_max": 0.05, "ray_resize_fps_min": 100.0}
PASS, FAIL, PENDING = "PASS", "FAIL", "NOT MEASURED"


def load_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def run_pytest() -> dict:
    """Run the whole suite and return counts plus one entry per test case."""
    JUNIT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", f"--junitxml={JUNIT}", "-q", "-o", "junit_family=xunit2"],
                          capture_output=True, text=True, check=False)
    cases = []
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        outcome, detail = "passed", ""
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                detail = (child.get("message") or "")[:200]
        cases.append({"file": case.get("classname", "").split(".")[-1], "name": case.get("name"),
                      "seconds": float(case.get("time") or 0), "outcome": outcome, "detail": detail})
    count = {k: sum(1 for c in cases if c["outcome"] == k) for k in ("passed", "failed", "error", "skipped")}
    return {"returncode": proc.returncode, "total": len(cases), **count, "cases": cases,
            "seconds": round(time.perf_counter() - t0, 1), "tail": proc.stdout.strip().splitlines()[-1:]}


def run_lint() -> dict:
    """PEP 8 check with ruff. `violations` is None when ruff is not installed."""
    try:
        proc = subprocess.run([sys.executable, "-m", "ruff", "check", ".", "--output-format", "concise"],
                              capture_output=True, text=True, check=False)
    except OSError:
        return {"violations": None}
    if "No module named ruff" in proc.stderr:
        return {"violations": None}
    lines = [ln for ln in proc.stdout.splitlines() if ln and ":" in ln and not ln.startswith(("Found", "All checks"))]
    return {"violations": 0 if proc.returncode == 0 else len(lines), "first": lines[:5]}


def criterion(cid, text, measured, ok, source) -> dict:
    status = PENDING if ok is None else (PASS if ok else FAIL)
    return {"id": cid, "criterion": text, "measured": measured, "status": status, "source": source}


def below(value, limit):
    return None if value is None else value <= limit


def above(value, limit):
    return None if value is None else value >= limit


def offline_criteria(tests: dict, lint: dict, th: dict) -> list:
    """Criteria judged from the test run and the pipeline reports in the repo."""
    prep, pre, split = (load_json(f"reports/{n}") for n in ("prepare_report.json", "preprocess_report.json",
                                                             "split_report.json"))
    exp = load_json("reports/export_report.json")
    f1, p95, diff = exp.get("onnx_test_macro_f1"), exp.get("latency_ms_p95"), exp.get("parity_max_abs_diff")
    ray_fps = (pre.get("throughput_fps") or {}).get("ray_resize")
    missing = split.get("classes_missing_from_val_or_test")
    bad = prep.get("bad_share")
    fmt = lambda v, spec: "not measured yet" if v is None else format(v, spec)  # noqa: E731
    return [
        criterion("AC-1", "Every automated test passes",
                  f"{tests['passed']} of {tests['total']} passed, {tests['failed'] + tests['error']} failed",
                  tests["failed"] + tests["error"] == 0 and tests["total"] > 0, "pytest"),
        criterion("AC-2", "The code follows PEP 8 (ruff reports no violation)",
                  "ruff not installed" if lint["violations"] is None else f"{lint['violations']} violations",
                  None if lint["violations"] is None else lint["violations"] == 0, "ruff check ."),
        criterion("AC-3", f"Test macro-F1 of the latest trained model (ONNX) is at least {th['macro_f1_min']:.2f}",
                  fmt(f1, ".3f"), above(f1, th["macro_f1_min"]), "reports/export_report.json"),
        criterion("AC-4", f"One prediction on 2 CPU threads takes at most {th['latency_ms_p95_max']:.0f} ms (p95)",
                  fmt(p95, ".1f") + (" ms" if p95 is not None else ""), below(p95, th["latency_ms_p95_max"]),
                  "reports/export_report.json"),
        criterion("AC-5", f"ONNX and PyTorch outputs differ by less than {th['parity_max_abs_diff']}",
                  fmt(diff, ".2e"), None if diff is None else diff < th["parity_max_abs_diff"],
                  "reports/export_report.json"),
        criterion("AC-6", "No recording session is shared between train, validation and test",
                  "enforced by tests/test_split.py", next((c["outcome"] == "passed" for c in tests["cases"]
                                                           if c["name"] == "test_no_session_in_two_splits"), None),
                  "pytest"),
        criterion("AC-7", "Every class has frames in the validation and the test split",
                  "not measured yet" if missing is None else ("all classes present" if not missing
                                                              else "missing: " + ", ".join(missing)),
                  None if missing is None else not missing, "reports/split_report.json"),
        criterion("AC-8", f"At most {th['data_bad_share_max']:.0%} of the raw rows fail the data quality checks",
                  fmt(bad, ".2%"), below(bad, th["data_bad_share_max"]), "reports/prepare_report.json"),
        criterion("AC-9", f"The Ray resize step handles at least {th['ray_resize_fps_min']:.0f} frames per second",
                  fmt(ray_fps, ".0f") + (" frames/s" if ray_fps is not None else ""),
                  above(ray_fps, th["ray_resize_fps_min"]), "reports/preprocess_report.json"),
    ]


def live_criteria(api: str, th: dict) -> list:
    """Criteria judged against the running stack on nova (needs the tailnet)."""
    import cv2
    import numpy as np
    import requests

    out, api = [], api.rstrip("/")

    def get(path):
        r = requests.get(f"{api}/{path}", timeout=15)
        r.raise_for_status()
        return r

    try:
        h = get("health").json()
        out.append(criterion("AC-10", "The API is up and serving a model", f"model v{h.get('model_version')}",
                             bool(h.get("model_loaded")), f"GET {api}/health"))
    except requests.RequestException as e:
        return [criterion("AC-10", "The API is up and serving a model", f"unreachable: {e}"[:120], False, api)]
    jpeg = cv2.imencode(".jpg", np.full((480, 640, 3), 120, dtype=np.uint8))[1].tobytes()
    times = []
    try:
        for _ in range(30):
            t0 = time.perf_counter()
            files = {"file": ("f.jpg", jpeg, "image/jpeg")}
            requests.post(f"{api}/predict", files=files, timeout=30).raise_for_status()
            times.append((time.perf_counter() - t0) * 1000)
        p95 = float(np.percentile(times, 95))
        out.append(criterion("AC-11", f"POST /predict answers within {th['api_latency_ms_p95_max']:.0f} ms "
                             "(p95 of 30 requests, network included)", f"{p95:.0f} ms",
                             p95 <= th["api_latency_ms_p95_max"], f"POST {api}/predict"))
    except requests.RequestException as e:
        out.append(criterion("AC-11", "POST /predict answers in time", f"failed: {e}"[:120], False, api))
    checks = [("AC-12", "Prometheus metrics are exposed", "metrics", lambda r: "roomsense_drift_ood_share" in r.text),
              ("AC-13", "The pipeline console answers and MLflow is reachable", "pipeline",
               lambda r: r.json().get("mlflow_up") is True),
              ("AC-14", "The latest pipeline run succeeded", "pipeline",
               lambda r: bool((r.json().get("latest") or {}).get("ok")))]
    for cid, text, path, test in checks:
        try:
            ok = bool(test(get(path)))
            out.append(criterion(cid, text, "yes" if ok else "no", ok, f"GET {api}/{path}"))
        except (requests.RequestException, ValueError) as e:
            out.append(criterion(cid, text, f"failed: {e}"[:120], False, f"GET {api}/{path}"))
    return out


def write_report(tests: dict, lint: dict, criteria: list, api: str) -> None:
    by_file = {}
    for c in tests["cases"]:
        f = by_file.setdefault(c["file"], {"passed": 0, "failed": 0, "error": 0, "skipped": 0})
        f[c["outcome"]] += 1
    measured = [c for c in criteria if c["status"] != PENDING]
    met = sum(1 for c in measured if c["status"] == PASS)
    mark = {"passed": "✅ pass", "failed": "❌ FAIL", "error": "❌ ERROR", "skipped": "⏭ skipped"}
    icon = {PASS: "✅ met", FAIL: "❌ not met", PENDING: "⏳ not measured"}
    lines = [
        "# RoomSense test report",
        "",
        "_Generated by `python scripts/test_report.py`. Do not edit by hand; rerun the script._",
        "",
        "| | |",
        "|---|---|",
        f"| Date | {datetime.now().astimezone():%d %b %Y, %H:%M (UTC%z)} |",
        f"| Code version (Git commit) | `{git_commit(short=True) or 'unknown'}` |",
        f"| Machine | {platform.node()} · {platform.system()} {platform.machine()} "
        f"· Python {platform.python_version()} |",
        f"| Live checks against | {api or 'not run (pass `--api` to include them)'} |",
        "",
        "## 1. Summary",
        "",
        "| Test cases | Passed | Failed | Errors | Skipped | Duration |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| **{tests['total']}** | **{tests['passed']}** | **{tests['failed']}** | {tests['error']} "
        f"| {tests['skipped']} | {tests['seconds']} s |",
        "",
        f"Acceptance criteria: **{met} of {len(measured)} measured criteria met**"
        + (f", {len(criteria) - len(measured)} not measured yet." if len(criteria) != len(measured) else "."),
        "",
        "## 2. Acceptance criteria",
        "",
        "Thresholds are defined in `params.yaml` under `acceptance:` and explained in the test plan "
        "(`docs/04-test-plan.md`, section 5).",
        "",
        "| ID | Criterion | Measured | Result | Evidence |",
        "|---|---|---|---|---|",
        *[f"| {c['id']} | {c['criterion']} | {c['measured']} | {icon[c['status']]} | `{c['source']}` |"
          for c in criteria],
        "",
        "## 3. Results per test file",
        "",
        "| Test file | Passed | Failed | Errors | Skipped |",
        "|---|---:|---:|---:|---:|",
        *[f"| `tests/{name}.py` | {f['passed']} | {f['failed']} | {f['error']} | {f['skipped']} |"
          for name, f in sorted(by_file.items())],
        "",
        "## 4. Every test case",
        "",
        "| # | Test file | Test case | Result | Time |",
        "|---:|---|---|---|---:|",
        *[f"| {i} | `{c['file']}` | `{c['name']}` | {mark[c['outcome']]}{' : ' + c['detail'] if c['detail'] else ''} "
          f"| {c['seconds'] * 1000:.0f} ms |" for i, c in enumerate(tests["cases"], 1)],
        "",
        "## 5. Code style",
        "",
        ("`ruff check .` was not run (ruff is not installed)." if lint["violations"] is None else
         f"`ruff check .` (PEP 8, import order, naming, likely bugs): **{lint['violations']} violations**."),
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tests, check acceptance criteria, write docs/05-test-report.md.")
    ap.add_argument("--api", default="", help="base URL of the running API for the live checks")
    a = ap.parse_args()
    th = dict(DEFAULTS)
    if Path("params.yaml").exists():
        th.update((yaml.safe_load(Path("params.yaml").read_text()) or {}).get("acceptance") or {})

    tests = run_pytest()
    lint = run_lint()
    criteria = offline_criteria(tests, lint, th)
    if a.api:
        criteria += live_criteria(a.api, th)
    ACCEPT.parent.mkdir(exist_ok=True)
    ACCEPT.write_text(json.dumps({"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                  "git_commit": git_commit(), "thresholds": th,
                                  "tests": {k: v for k, v in tests.items() if k != "cases"},
                                  "criteria": criteria}, indent=2))
    write_report(tests, lint, criteria, a.api)
    failed = [c["id"] for c in criteria if c["status"] == FAIL]
    log.info("%d tests: %d passed, %d failed. Criteria not met: %s. Report: %s", tests["total"], tests["passed"],
             tests["failed"] + tests["error"], failed or "none", REPORT)
    return 1 if failed or tests["returncode"] not in (0,) else 0


if __name__ == "__main__":
    sys.exit(main())
