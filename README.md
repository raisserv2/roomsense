# RoomSense

Classifies the state of a hostel room (`at_desk`, `lying_in_bed`, `sitting_on_bed`, `in_room`, `empty`, `lights_off`) from a phone camera, with the full MLOps loop around it: versioned data, a reproducible pipeline, a model registry, CPU serving, monitoring, drift alerts and retraining. Everything runs on my own machines. Nothing runs in the cloud.

```
 Honor 7X (IP Webcam) ──frames──▶ capture service ──labelled frames──▶ data/raw (DVC → Seagate on nova)
                                         │ active session label = ground truth
                                         ▼
 ┌─────────────────────────── nova (ThinkPad T460, Docker) ───────────────────────────┐
 │  api (FastAPI + ONNX Runtime) ◀── champion model ── mlflow (tracking + registry)  │
 │    │ camera loop: predict every 5 s, drift window, live accuracy                    │
 │    ├── /metrics ──▶ prometheus ──alerts──▶ alertmanager ──▶ Telegram               │
 │    │                     │                        └──webhook──▶ api /hooks (retrain flag)
 │    │                     └──▶ grafana (dashboard)                                   │
 │    └── frontend (nginx: live state, drift, model, upload)                           │
 └─────────────────────────────────────────────────────────────────────────────────────┘
                                         ▲ register + promote          │ retrain flag
 ┌──────────────── Legion (RTX 4060) ────┴─────────────────────────────▼──────────────┐
 │  dvc repro: prepare → preprocess (Ray) → split → train (MLflow) → export → register │
 │  scripts/retrain.py: polls the flag → dvc pull/repro/push → git push → ack          │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

All web UIs are reachable only inside the tailnet (Tailscale Serve, HTTPS). Every Docker port is bound to `127.0.0.1`.

## The cogs

| Cog | Where | Input → Output |
|---|---|---|
| `capture/` | nova | camera frames + label buttons → `data/raw/<session>/*.jpg`, `frames.csv`; review tool for curation |
| `prepare` | Legion (DVC) | `data/raw` → `manifest.csv` (missing files dropped, day/night added) |
| `preprocess` | Legion (DVC, Ray) | manifest → trimmed, resized 224×224, de-duplicated frames (k-means per session) |
| `split` | Legion (DVC) | clean manifest → `splits.csv`, split **by session** (no leakage), optional `data.include` filter |
| `train` | Legion (DVC, GPU) | splits → `models/best.pt`, MLflow run (params, curves, test macro-F1, PyTorch model) |
| `export` | Legion (DVC) | checkpoint → `model.onnx` + parity check + CPU latency + reference stats for drift |
| `register` | Legion (DVC) | ONNX → MLflow registry version; promoted to alias `champion` only if it beats the current champion **on the same test split** |
| `mlflow` | nova (Docker) | tracking server + model registry (sqlite + artifact store) |
| `api` | nova (Docker) | champion model (hot-reloaded) + camera frames → predictions, `/metrics`, drift + live accuracy |
| `frontend` | nova (Docker) | separate nginx container: live frame, room state, drift monitor, model card, photo upload |
| `prometheus` + `alertmanager` | nova (Docker) | metrics → alerts (drift, confidence, accuracy, API/camera down) → Telegram + retrain webhook |
| `grafana` | nova (Docker) | provisioned "RoomSense: live model monitoring" dashboard |
| `retrain.py` | Legion | retrain flag → `dvc repro` → new champion → ack |

## Drift → retrain, in one paragraph

Every camera frame's brightness goes into a rolling window (120 frames, about 10 minutes). The API compares the window with the training frames: the **unseen-lighting share** is the share of recent frames whose brightness falls in a range the model (almost) never saw in training. When it stays above 50 % for 5 minutes, Prometheus fires `RoomSenseDrift`. Alertmanager notifies Telegram and calls the API webhook, which raises the **retrain flag**. On the Legion, `scripts/retrain.py` sees the flag, pulls the newest nightly snapshot, reruns the pipeline, and the `register` stage promotes the new model if it beats the old one on the same test split. The API hot-reloads the new champion within a minute. Its reference statistics now cover the new lighting, so the drift signal drops back. PSI is also computed and shown, but it isn't used for alerting, because on a short window it is high even without drift. While a capture session is recording, its label is used as ground truth, so the dashboard also shows **live accuracy**.

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
pytest                                   # 28 tests
echo 'export MLFLOW_TRACKING_URI=https://nova.tailb5cb87.ts.net:5443' >> ~/.bashrc && source ~/.bashrc
dvc repro                                # trains, exports, registers + promotes on nova
dvc push && git add dvc.lock reports && git commit -m "pipeline run" && git push
```

Within a minute, the UI at `https://nova.tailb5cb87.ts.net:4443` shows the live room state.

### Day-to-day
| Task | Command |
|---|---|
| Rerun only what changed | `dvc repro` |
| Pipeline DAG | `dvc dag` |
| Compare runs | MLflow UI (`:5443`) or `dvc metrics show` |
| Retrain when drift is flagged | `python scripts/retrain.py` (or `--watch 10` to keep polling) |
| Retrain now | `python scripts/retrain.py --force` |
| Run via MLproject | `mlflow run . -e main --env-manager local` (also `-e train -P epochs=3`) |
| MLflow's own model server | `docker compose --profile mlflow-serve up -d mlflow-serve`, then `python scripts/mlflow_serve_request.py photo.jpg` |
| Tests | `pytest` |

## API

`GET /health` · `GET /model` · `POST /predict` (multipart `file`, `?monitor_frame=true` feeds the drift window) · `GET /latest` · `GET /latest.jpg` · `GET /drift` · `POST /reload` · `POST /hooks/alertmanager` · `GET /retrain` · `POST /retrain/request` · `POST /retrain/ack` · `GET /metrics`. Interactive docs at `/api/docs` on the UI host.

## Privacy

Frames never leave the room network and my own drives (Tailscale only, Funnel off, no cloud). The camera is covered when guests are in. Face blurring with a Haar detector was tried in `preprocess`, found unreliable (missed faces, blurred furniture), and is switched off (`blur_faces: false`). A stronger detector is future work.
