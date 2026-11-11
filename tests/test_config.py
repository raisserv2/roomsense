"""Deployment configuration checks: loose coupling, monitoring coverage, alert rules."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def test_frontend_reaches_the_backend_only_through_configuration():
    """No page or script may hard-code where the API lives."""
    site = DEPLOY / "frontend" / "site"
    for f in [*site.glob("*.js"), *site.glob("*.html")]:
        if f.name == "config.js":
            continue
        text = f.read_text()
        assert "api:8000" not in text and "8300" not in text, f.name
        assert not re.search(r"fetch\(\s*[\"']", text), f"{f.name} calls fetch() with a literal URL"
    template = (DEPLOY / "frontend" / "default.conf.template").read_text()
    assert "proxy_pass ${API_UPSTREAM}/;" in template
    compose = yaml.safe_load((DEPLOY / "compose.yaml").read_text())
    env = compose["services"]["frontend"]["environment"]
    assert "API_UPSTREAM" in env and "API_BASE_URL" in env


def test_frontend_and_backend_are_separate_compose_services():
    services = yaml.safe_load((DEPLOY / "compose.yaml").read_text())["services"]
    assert services["frontend"]["build"] != services["api"]["build"]
    assert {"mlflow", "api", "frontend", "prometheus", "alertmanager", "grafana"} <= set(services)


def test_every_port_is_bound_to_localhost():
    services = yaml.safe_load((DEPLOY / "compose.yaml").read_text())["services"]
    for name, svc in services.items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} beyond localhost"


def test_prometheus_scrapes_every_component():
    services = set(yaml.safe_load((DEPLOY / "compose.yaml").read_text())["services"])
    prom = yaml.safe_load((DEPLOY / "prometheus" / "prometheus.yml").read_text())
    jobs = {j["job_name"]: j["static_configs"][0]["targets"][0].split(":")[0] for j in prom["scrape_configs"]}
    assert {"roomsense-api", "roomsense-capture", "frontend", "node", "cadvisor", "prometheus", "alertmanager",
            "grafana"} <= set(jobs)
    for job, host in jobs.items():
        assert host in services | {"localhost", "roomsense-capture"}, f"{job} scrapes unknown host {host}"


def test_alert_rules_are_well_formed_and_drift_requests_retrain():
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text())["groups"][0]["rules"]
    names = [r["alert"] for r in rules]
    assert len(names) == len(set(names)) and len(names) >= 10
    for r in rules:
        assert r["expr"] and r["labels"]["severity"] in {"warning", "critical"} and r["annotations"]["summary"]
    drift = next(r for r in rules if r["alert"] == "RoomSenseDrift")
    assert drift["labels"]["action"] == "retrain" and "roomsense_drift_ood_share" in drift["expr"]


def test_alert_metrics_are_actually_exported():
    """Every roomsense_* metric used in an alert or the dashboard must exist in the code."""
    defined = set()
    for src in (DEPLOY / "api" / "app" / "metrics.py", ROOT / "capture" / "capmetrics.py"):
        defined |= set(re.findall(r'\("(roomsense_[a-z_]+)"', src.read_text()))
    used = set(re.findall(r"roomsense_[a-z_]+", (DEPLOY / "prometheus" / "alerts.yml").read_text()))
    used |= set(re.findall(r"roomsense_[a-z_]+", (DEPLOY / "grafana" / "dashboards" / "roomsense.json").read_text()))
    suffixes = ("_bucket", "_total", "_count", "_sum")
    for metric in used:
        base = next((metric[:-len(s)] for s in suffixes if metric.endswith(s)), metric)
        assert metric in defined or base in defined or base + "_total" in defined, metric


def test_dvc_stages_and_mlproject_entry_points_exist():
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    assert list(stages) == ["prepare", "preprocess", "split", "train", "export", "register"]
    project = yaml.safe_load((ROOT / "MLproject").read_text())
    assert project["python_env"] == "python_env.yaml" and (ROOT / "python_env.yaml").exists()
    for name, entry in project["entry_points"].items():
        script = re.search(r"python (scripts/\w+\.py)", entry["command"])
        assert script and (ROOT / script.group(1)).exists(), name
