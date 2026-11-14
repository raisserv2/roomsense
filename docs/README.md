# RoomSense documentation

| # | Document | What it is |
|---|---|---|
| 1 | [Architecture](01-architecture.md) | The architecture diagram with an explanation of every block, plus pipeline, serving, drift-sequence and deployment diagrams |
| 2 | [High-Level Design](02-hld.md) | Design choices and the reasons for them |
| 3 | [Low-Level Design](03-lld.md) | Every API endpoint with input/output specification, modules and classes, metrics, file formats, configuration |
| 4 | [Test plan and test cases](04-test-plan.md) | Levels, cases, acceptance criteria |
| 5 | [Test report](05-test-report.md) | Generated: counts, every case, criteria met |
| 6 | [User manual](06-user-manual.md) | For a non-technical user, with screenshots |

PDF copies are in [`pdf/`](pdf/). Diagrams (SVG, PNG, PDF) and the script that draws them are in [`diagrams/`](diagrams/); icon credits in [`diagrams/CREDITS.md`](diagrams/CREDITS.md).

## Where each requirement is covered

### Demonstration

| Requirement | Where |
|---|---|
| Intuitive UI for a non-technical user, foolproof, no UI errors, responsive | `deploy/frontend/site/` · Live room page with plain-language states, empty/error states, upload validation · test cases TC-UI-1 to TC-UI-9 |
| User manual | [06-user-manual.md](06-user-manual.md) and the in-app Help page |
| Separate UI screen for the ML pipeline, including data ingestion | Pipeline page (`pipeline.html`): DAG, per-stage status, "Pictures collected" |
| Pipeline management console | Pipeline page → Controls (retrain, reload, rollback) and Model versions |
| Console of errors, failures and successful runs | Pipeline page → Runs (✓/✗ per run, failed stage and error) and Activity |
| Speed and throughput of the pipeline | Pipeline page tiles and per-stage figures · Grafana "Training pipeline" row · `reports/preprocess_report.json` |

### Software engineering

| Requirement | Where |
|---|---|
| Design document, paradigm | [02-hld.md](02-hld.md) section 3 |
| LLD with endpoint definitions and I/O specification | [03-lld.md](03-lld.md) sections 2 and 3 |
| Architecture diagram, HLD diagram | [01-architecture.md](01-architecture.md) |
| Loose coupling, configurable REST between frontend and backend | `deploy/frontend/default.conf.template`, `40-roomsense-config.sh`, `API_UPSTREAM` / `API_BASE_URL` · `tests/test_config.py` |
| PEP 8 | `pyproject.toml` (ruff) · AC-2 in the test report |
| Logging | `roomsense/runlog.py` (pipeline), `logging` in the API and capture · LLD section 9 |
| Exception handling | LLD section 9 · API error tests in `tests/test_api.py` |
| Inline documentation | Module and function docstrings throughout |
| Unit tests | `tests/` (91 cases) |
| Test plan, test cases, test report, acceptance criteria | [04-test-plan.md](04-test-plan.md), [05-test-report.md](05-test-report.md), `params.yaml: acceptance` |

### AIOps

| Requirement | Where |
|---|---|
| Data ingestion / transformation pipeline with Ray | `capture/` (ingestion), `scripts/prepare.py`, `scripts/preprocess.py` (`@ray.remote`) |
| Throughput and speed | `reports/preprocess_report.json` (`seconds`, `throughput_fps`), `reports/stages/*.json`, MLflow experiment `roomsense-pipeline` |
| CI with DVC, the DVC DAG | `dvc.yaml`, `scripts/run_pipeline.py` · `dvc dag` |
| Versioning with Git and DVC | `data/raw.dvc`, `dvc.lock`, `scripts/nightly-snapshot.sh` |
| Experiment tracking: metrics, params, artifacts; beyond autolog | `scripts/train.py` (manual logging: per-epoch curves, per-class F1, calibration, confusion matrix, provenance tags) |
| Prometheus instrumentation, information points | LLD section 6 (36 API metrics, 11 capture metrics) |
| All components monitored | `deploy/prometheus/prometheus.yml` (8 jobs), exporters in `deploy/compose.yaml` · `tests/test_config.py` |
| Grafana near real time | `deploy/grafana/dashboards/roomsense.json` (42 panels, 10 s refresh) |
| MLflow for API-fication | `mlflow.onnx.log_model` in `scripts/register.py`, `mlflow-serve` compose profile, `scripts/mlflow_serve_request.py` |
| MLprojects, identical dev and test environments | `MLproject`, `python_env.yaml`, entry point `test` |
| FastAPI for the UI's APIs | `deploy/api/app/main.py` |
| Dockerised backend and frontend, separate compose services | `deploy/api/Dockerfile`, `deploy/frontend/Dockerfile`, `deploy/compose.yaml` |

### Guideline extras

| Item | Where |
|---|---|
| Reproducible from a Git commit hash + MLflow run ID | Run tags `git_commit`, `data_version`; `dvc.lock` stored with each run |
| Data quality checks | `scripts/prepare.py`, `reports/prepare_report.json`, AC-8 |
| Rollback mechanism | `roomsense/registry.py`, `scripts/rollback.py`, `POST /pipeline/rollback`, Pipeline page button |
| Security, encryption | HLD section 6 |
| Explainability | HLD section 7 |
