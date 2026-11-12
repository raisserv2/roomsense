# RoomSense

Classifies the state of a hostel room (`at_desk`, `lying_in_bed`, `sitting_on_bed`, `in_room`, `empty`, `lights_off`) from a phone camera, with the full MLOps loop around it: versioned data, a reproducible pipeline, a model registry, CPU serving, monitoring, drift alerts and retraining. Everything runs on my own machines. Nothing runs in the cloud.

![RoomSense architecture](docs/diagrams/01-architecture.png)

**Documents** (in [`docs/`](docs/README.md)): [architecture](docs/01-architecture.md) · [high-level design](docs/02-hld.md) · [low-level design with the API specification](docs/03-lld.md) · [test plan](docs/04-test-plan.md) · [test report](docs/05-test-report.md) · [user manual](docs/06-user-manual.md)

All web UIs are reachable only inside the tailnet (Tailscale Serve, HTTPS). Every Docker port is bound to `127.0.0.1`.

## The cogs

| Cog | Where | Input → Output |
|---|---|---|
| `capture/` | nova | camera frames + label buttons → `data/raw/<session>/*.jpg`, `frames.csv`; review tool for curation |
| `prepare` | Legion (DVC) | `data/raw` → `manifest.csv` (missing files dropped, day/night added) |
| `preprocess` | Legion (DVC, Ray) | manifest → trimmed, resized 224×224, de-duplicated frames (k-means per session) |
| `split` | Legion (DVC) | clean manifest → `splits.csv`, split **by session** (no leakage), optional `data.include` filter |
| `train` | Legion (DVC, GPU) | splits → `models/best.pt` + `calibration.json`, MLflow run (params, curves, test macro-F1, confusion matrix, PyTorch model). Class/session-balanced sampling, label smoothing, frozen-head warm-up then fine-tune, colour augmentation, early stopping, temperature calibration |
| `export` | Legion (DVC) | checkpoint ÷ temperature → `model.onnx` (serves calibrated confidences) + parity check + CPU latency + reference stats for drift |
| `register` | Legion (DVC) | ONNX → MLflow registry version; promoted to alias `champion` only if it beats the current champion **on test sessions neither model trained on** |
| `mlflow` | nova (Docker) | tracking server + model registry (sqlite + artifact store) |
| `api` | nova (Docker) | champion model (hot-reloaded) + camera frames → predictions, `/metrics`, drift + live accuracy |
| `frontend` | nova (Docker) | separate nginx container: live frame, room state, drift monitor, model card, photo upload |
| `prometheus` + `alertmanager` | nova (Docker) | metrics → alerts (drift, confidence, accuracy, API/camera down) → Telegram + retrain webhook |
| `grafana` | nova (Docker) | provisioned "RoomSense: live model monitoring" dashboard |
| `retrain.py` | Legion | retrain flag → `dvc repro` → new champion → ack |

## Drift → retrain, in one paragraph

Every camera frame's brightness goes into a rolling window (120 frames, about 10 minutes). The API compares the window with the training frames: the **unseen-lighting share** is the share of recent frames whose brightness falls in a range the model (almost) never saw in training. When it stays above 50 % for 5 minutes, Prometheus fires `RoomSenseDrift`. Alertmanager notifies Telegram and calls the API webhook, which raises the **retrain flag**. On the Legion, `scripts/retrain.py` sees the flag, pulls the newest nightly snapshot, reruns the pipeline, and the `register` stage promotes the new model if it beats the old one on test sessions neither has trained on. The API hot-reloads the new champion within a minute. Its reference statistics now cover the new lighting, so the drift signal drops back. PSI is also computed and shown, but it isn't used for alerting, because on a short window it is high even without drift. While a capture session is recording, its label is used as ground truth, so the dashboard also shows **live accuracy**.

**Demo recipe:** set `data.include: {lighting: [tube_on]}` in `params.yaml`, `dvc repro`, and the champion now only knows the tube-light-on room. Switch the tube light off: within about 10 minutes the drift alert fires and a retrain is requested. Set `data.include: {}`, run `python scripts/retrain.py`, and the alert resolves.

## Setup

### nova (serving + monitoring)
```bash
cd ~/roomsense
cp -a data/trash ~/trash-backup          # git pull removes the trash files Git used to track
git pull
cp -an ~/trash-backup/. data/trash/      # put them back (now ignored by Git)

docker network create roomsense          # once: shared by capture and the serving stack
(cd capture && docker compose up -d)     # capture joins the shared network

cd deploy
cp .env.example .env && nano .env        # CAM_PASS, Telegram bot token + chat id, Grafana password
mkdir -p data/{mlflow,api,prometheus,alertmanager,grafana}
docker compose up -d --build
docker compose ps

sudo tailscale serve --bg --https=4443 http://127.0.0.1:8200   # RoomSense UI
sudo tailscale serve --bg --https=5443 http://127.0.0.1:5000   # MLflow
sudo tailscale serve --bg --https=3443 http://127.0.0.1:3300   # Grafana
sudo tailscale serve --bg --https=6443 http://127.0.0.1:9091   # Prometheus
```

### Legion (training)
```bash
cd ~/roomsense && source .venv/bin/activate
git stash                                # local edits are already in this version
git pull
uv pip install -r requirements.txt -r requirements-dev.txt "mlflow==3.17.2"
pytest                                   # 91 tests
ruff check .                             # PEP 8
echo 'export MLFLOW_TRACKING_URI=https://nova.tailb5cb87.ts.net:5443' >> ~/.bashrc && source ~/.bashrc
python scripts/run_pipeline.py           # dvc repro + records the run (status, timings, throughput) in MLflow
dvc push && git add dvc.lock reports && git commit -m "pipeline run" && git push
```

Within a minute, the UI at `https://nova.tailb5cb87.ts.net:4443` shows the live room state.

### Day-to-day
| Task | Command |
|---|---|
| Rerun only what changed, and record the run | `python scripts/run_pipeline.py` (plain `dvc repro` also works, without the record) |
| Pipeline DAG | `dvc dag` |
| Compare runs | MLflow UI (`:5443`) or `dvc metrics show` |
| Retrain when drift is flagged | `python scripts/retrain.py` (or `--watch 10` to keep polling) |
| Retrain now | `python scripts/retrain.py --force` |
| Roll back to the previous model | `python scripts/rollback.py` (`--list`, `--to 2`), or the button on the Pipeline page |
| Run via MLproject | `mlflow run . -e main --env-manager local` (also `-e train -P epochs=3`) |
| Same environment for dev and test | `mlflow run . -e test` builds the env from `python_env.yaml` and runs the tests in it |
| MLflow's own model server | `docker compose --profile mlflow-serve up -d mlflow-serve`, then `python scripts/mlflow_serve_request.py photo.jpg` |
| Tests | `pytest` |
| Tests + lint + acceptance criteria → `docs/05-test-report.md` | `python scripts/test_report.py` (add `--api https://nova.tailb5cb87.ts.net:4443/api` for the live checks) |
| Rebuild diagrams / PDFs | `python docs/diagrams/make_diagrams.py` · `python docs/build_pdfs.py` |

## API

Full specification with inputs, outputs and error codes: [docs/03-lld.md](docs/03-lld.md). Interactive docs at `/api/docs` on the UI host.

Live: `GET /health` · `GET /model` · `POST /predict` · `GET /latest` · `GET /latest.jpg` · `GET /drift`
Pipeline console: `GET /pipeline` · `GET /pipeline/runs` · `GET /pipeline/registry` · `POST /pipeline/rollback` · `GET /events`
Lifecycle: `POST /reload` · `POST /hooks/alertmanager` · `GET /retrain` · `POST /retrain/request` · `POST /retrain/ack`
Monitoring: `GET /metrics`

## Web UI

Three pages at `https://nova.tailb5cb87.ts.net:4443`: **Live room** (current state, confidence, drift, photo upload), **Pipeline** (the DAG with status and throughput per stage, controls for retrain / reload / rollback, data ingestion totals, every successful and failed run, model versions, activity log) and **Help** (the user manual). The frontend reaches the backend only through `API_UPSTREAM` and `API_BASE_URL` in `deploy/.env`.

## Monitoring

Prometheus scrapes eight jobs: the API, the capture service, the frontend (nginx exporter), the host (node-exporter), all containers (cAdvisor), and Prometheus, Alertmanager and Grafana themselves. 13 alert rules; drift and live-accuracy alerts carry `action=retrain`. The Grafana dashboard has rows for the model, the training pipeline, data ingestion, and components and host.

## Privacy

Frames never leave the room network and my own drives (Tailscale only, Funnel off, no cloud). The camera is covered when guests are in. Face blurring with a Haar detector was tried in `preprocess`, found unreliable (missed faces, blurred furniture), and is switched off (`blur_faces: false`). A stronger detector is future work.
