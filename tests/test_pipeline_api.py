"""Pipeline console endpoints, with a fake MLflow behind the API."""
import pytest

from tests.conftest import FakeMlflow, FakeRun, FakeVersion

STAGES = ["prepare", "preprocess", "split", "train", "export", "register"]


def pipeline_run(run_id, ok=True, start=1_760_000_000_000):
    tags = {"trigger": "drift", "reason": "RoomSenseDrift", "git_commit": "a" * 40, "data_version": "d" * 32,
            **{f"stage_{s}_status": "ok" for s in STAGES}}
    metrics = {"total_seconds": 312.0, "stage_preprocess_seconds": 80.0, "stage_preprocess_items": 27000.0,
               "stage_preprocess_throughput": 337.5, "ray_resize_fps": 1250.0, "test_macro_f1": 0.71,
               "latency_ms_p95": 2.4}
    if not ok:
        tags.update(stage_train_status="failed", stage_export_status="not_reached",
                    stage_register_status="not_reached", failed_stage="train", error="CUDA out of memory")
    return FakeRun(run_id, "FINISHED" if ok else "FAILED", tags, metrics, start=start)


@pytest.fixture
def console(client):
    """Swap the API's MLflow client and capture-stats source for fakes; restore afterwards."""
    from app import main
    fake = FakeMlflow(
        [FakeVersion(1, {"decision": "promoted: no champion yet", "test_macro_f1": "0.55"}),
         FakeVersion(2, {"decision": "promoted: better", "test_macro_f1": "0.71", "previous_champion": "1"})],
        champion=2,
        runs={"roomsense-pipeline": [pipeline_run("p3", ok=False, start=1_760_000_300_000), pipeline_run("p2"),
                                     pipeline_run("p1", start=1_759_999_000_000)],
              "roomsense": [FakeRun("t1", "FINISHED", {"mlflow.runName": "train", "git_commit": "b" * 40},
                                    {"test_macro_f1": 0.71}, {"train_frames": "900"})]})
    store = main.pipeline
    old = (store._client_factory, store._stats_fetcher)
    store._client_factory = lambda: fake
    store._stats_fetcher = lambda: {"available": True, "frames_total": 27000, "sessions_total": 80,
                                    "per_class": {"at_desk": {"frames": 9000, "sessions": 20}}}
    store.refresh()
    yield client, fake
    store._client_factory, store._stats_fetcher = old
    store.refresh()


def test_overview_without_mlflow_still_answers(client):
    from app import main
    main.pipeline.refresh()
    r = client.get("/pipeline").json()
    assert r["mlflow_up"] is False and r["errors"]
    assert [s["name"] for s in r["dag"]["stages"]] == STAGES     # the static DAG
    assert r["latest"] is None and r["summary"]["runs_total"] == 0


def test_overview_shows_dag_latest_run_and_throughput(console):
    client, _ = console
    r = client.get("/pipeline").json()
    assert r["mlflow_up"] and r["errors"] == []
    assert r["latest"]["run_id"] == "p3" and r["latest"]["ok"] is False and r["latest"]["failed_stage"] == "train"
    assert r["last_success"]["run_id"] == "p2"
    assert r["last_success"]["stages"]["preprocess"]["throughput_per_s"] == 337.5
    assert r["last_success"]["headline"]["ray_resize_fps"] == 1250.0
    assert r["summary"] == {**r["summary"], "runs_total": 3, "runs_ok": 2, "runs_failed": 1, "champion": "2"}
    assert r["ingestion"]["frames_total"] == 27000
    assert {"capture", "serve", *STAGES} <= set(r["stage_info"])
    assert "requested" in r["retrain"]


def test_runs_console_lists_failures_and_successes(console):
    client, _ = console
    r = client.get("/pipeline/runs?limit=2").json()
    assert [x["run_id"] for x in r["pipeline_runs"]] == ["p3", "p2"]
    assert r["pipeline_runs"][0]["status"] == "FAILED" and r["pipeline_runs"][0]["error"] == "CUDA out of memory"
    assert r["training_runs"][0]["test_macro_f1"] == 0.71 and r["training_runs"][0]["git_commit"] == "b" * 40
    assert client.get("/pipeline/runs?limit=0").status_code == 422


def test_registry_lists_versions_and_rollback_target(console):
    client, _ = console
    r = client.get("/pipeline/registry").json()
    assert r["champion"] == "2" and r["rollback_target"] == "1" and r["model"] == "roomsense"
    assert [v["version"] for v in r["versions"]] == ["2", "1"]


def test_rollback_moves_the_alias_and_is_logged(console):
    client, fake = console
    r = client.post("/pipeline/rollback")
    assert r.status_code == 200 and r.json()["champion"] == "1" and r.json()["rolled_back_from"] == "2"
    assert fake.aliases["champion"] == "1"
    assert client.get("/pipeline/registry").json()["champion"] == "1"
    assert any(e["kind"] == "rollback" for e in client.get("/events").json()["events"])
    # nothing older than v1: a second rollback is refused with a clear message
    again = client.post("/pipeline/rollback")
    assert again.status_code == 409 and "No earlier version" in again.json()["detail"]


def test_rollback_needs_the_token_when_one_is_set(console, monkeypatch):
    client, fake = console
    from app import settings
    monkeypatch.setattr(settings, "RETRAIN_TOKEN", "s3cret")
    assert client.post("/pipeline/rollback").status_code == 403
    assert client.post("/retrain/request").status_code == 403
    assert client.post("/pipeline/rollback", headers={"X-Token": "s3cret"}).status_code == 200


def test_rollback_without_mlflow(client):
    assert client.post("/pipeline/rollback").status_code == 503


def test_pipeline_metrics_reach_prometheus(console):
    client, _ = console
    text = client.get("/metrics").text
    assert 'roomsense_pipeline_stage_throughput{stage="preprocess"} 337.5' in text
    assert "roomsense_pipeline_last_run_success 0.0" in text
    assert 'roomsense_pipeline_runs{result="failed"} 1.0' in text
    assert 'roomsense_dependency_up{dependency="mlflow"} 1.0' in text
    assert "roomsense_http_requests_total" in text


def test_activity_log_records_alerts(client):
    alert = {"alerts": [{"status": "firing", "labels": {"alertname": "RoomSenseCameraStale"},
                         "annotations": {"summary": "No camera frame for 3m"}}]}
    client.post("/hooks/alertmanager", json=alert)
    events = client.get("/events?limit=5").json()["events"]
    assert events[0]["kind"] == "alert" and "RoomSenseCameraStale firing" in events[0]["message"]
    assert client.get("/events?limit=999").status_code == 422
