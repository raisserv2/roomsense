# RoomSense: High-Level Design (HLD)

This document explains the design choices and why each was made. The diagrams and the block-by-block description are in [01-architecture.md](01-architecture.md); endpoint and module detail is in [03-lld.md](03-lld.md).

## 1. Problem and goals

**Problem.** Tell, from a fixed phone camera, which of six states a hostel room is in (`at_desk`, `empty`, `in_room`, `lights_off`, `lying_in_bed`, `sitting_on_bed`), continuously, and keep doing it well when conditions change.

**Goals, in priority order.**

1. A complete, reproducible machine learning lifecycle: versioned data, a pipeline, tracked experiments, a registry, serving, monitoring, drift detection, retraining, rollback.
2. A web UI that a non-technical person can read, and a screen that shows and manages the pipeline.
3. Everything on local hardware. The course forbids cloud, and the pictures are of a private room.
4. A model good enough to make the loop meaningful. Accuracy is not the main objective.

**Non-goals.** Recognising who is in the room, working in other rooms or from other camera angles, multi-user access control.

## 2. Constraints that shaped the design

| Constraint | Consequence |
|---|---|
| No cloud | Self-hosted MLflow, DVC remote on a USB drive, Tailscale for remote access instead of a public server |
| nova has no GPU and 8 GB RAM | Training on the Legion; serving with ONNX Runtime on CPU; a small model; SQLite instead of a database server |
| The Legion is not always on | Retraining is pull-based: the Legion polls a flag when it is on. nova never needs to reach the Legion. |
| Private images | Nothing image-like goes to GitHub; pages open only inside the tailnet; frames are stored only during sessions the owner starts |
| Small dataset from one room (about 27 000 raw frames, about 80 sessions) | Transfer learning; session-level splitting; de-duplication; calibration; a metric that tolerates missing classes |
| One developer, a few weeks | Standard tools over custom ones; one shared Python library for training and serving |

## 3. Paradigm and structure

The code is mostly **functional**: modules of small functions with explicit inputs and outputs (`roomsense/frames.py`, `drift.py`, `registry.py`, `evalmetrics.py`), which is what makes them unit-testable without a camera, a GPU or a tracking server. **Classes are used only where there is state with a lifecycle**: `ModelManager` (the loaded model and its hot swap), `Monitor` (the drift window), `PipelineStore` (a cached snapshot), `RetrainState` and `EventLog` (small persistent files), `StageRun` (one timed stage).

The repository has four parts with one-way dependencies:

- `roomsense/`: the shared library. Depends on nothing else in the repo.
- `scripts/`: the DVC stages and operations. Import `roomsense/`.
- `deploy/`: the serving stack. The API imports `roomsense/`; the frontend imports nothing and only calls REST.
- `capture/`: data ingestion. Stands alone (its own compose file) so recording keeps working while the serving stack is rebuilt.

## 4. Design decisions and rationale

### 4.1 Data

**Label at capture time, in sessions.** The owner taps a label and the service records until stopped. Labelling 27 000 frames afterwards would not have happened. A session is also the natural unit for splitting and for privacy: nothing is saved unless a session is running.

**Split by session, never by frame.** Consecutive frames 5 s apart are nearly identical. A random frame split would put near-copies in train and test and report inflated scores. `split.py` assigns whole sessions and balances classes across splits with random restarts and greedy moves. A test proves no session is in two splits.

**De-duplicate inside each session with k-means on embeddings.** A 10-minute session is 120 frames of almost the same picture. Keeping 5 to 80 representative frames per session (nearest to cluster centres) stops long sessions from dominating and shortens training. Alternative considered: keep every n-th frame. Rejected because it keeps duplicates of a still scene and can drop the one moment something changed.

**Ray for preprocessing.** Resizing and per-session clustering are independent tasks, so they parallelise across CPU cores as Ray tasks. Spark was the alternative; it needs a JVM and a cluster abstraction that brings nothing on one machine, while Ray is `pip install` and the same code would scale to more nodes. Throughput (frames per second) is measured per step and reported.

**Data quality gate.** `prepare.py` checks every row (file exists and is not empty, known label and lighting, possible combination, valid timestamp, no duplicates) and fails the pipeline if more than 5% are bad. A broken snapshot should stop the run, not train a model.

**Face blurring was tried and switched off.** Haar cascades missed faces in profile and in dim light and blurred random patches. A filter that works sometimes gives false comfort. Privacy is instead handled by never letting images leave the two machines (section 6).

### 4.2 Versioning and continuous integration

**Git for code, DVC for data and models.** Git stores `dvc.yaml`, `dvc.lock`, `params.yaml` and `data/raw.dvc`, small text files that pin the exact data and outputs by hash. DVC stores the heavy files on the Seagate. One Git commit therefore identifies code + data + parameters.

**The DVC DAG is the CI (continuous integration) system.** `dvc repro` reruns only stages whose inputs or parameters changed and fails fast on the first broken stage. With no cloud, a hosted CI service was not an option; a DAG that rebuilds and re-validates on change is the local equivalent. `scripts/run_pipeline.py` wraps it and records every run, including failures.

**Reproducibility by commit + run ID.** Each MLflow run carries the tags `git_commit`, `git_dirty` and `data_version`, and stores `dvc.lock` and `params.yaml` as artifacts.

### 4.3 Model and training

**MobileNetV3-Small with transfer learning.** About 1.5 million parameters, a 6 MB ONNX file, a few milliseconds per frame on CPU. Larger networks would overfit 900-odd training frames and slow nova down for no gain.

**No flips, no crops.** The desk is at the right edge of the frame and the bed on the left. Position is the signal; flipping or cropping would destroy it. Only colour jitter is used.

**Class- and session-balanced sampling, label smoothing.** `at_desk` has many times more frames than `lights_off`. A weighted sampler evens out classes and sessions; label smoothing reduces overconfidence.

**Temperature calibration.** The first model was 99% confident when wrong. One scalar fitted on the validation set and baked into the ONNX graph makes confidence usable in the UI and in the low-confidence alert. Expected Calibration Error is logged before and after.

**Macro-F1 over the classes present in the split.** With few `lights_off` sessions the test split can have none. Averaging a zero for a class with no test frames measures the split, not the model. `evalmetrics.macro_f1_present` averages over classes that occur and reports which were skipped. The candidate and the champion are always compared with the same function on the same split.

### 4.4 Packaging and serving

**ONNX Runtime, not PyTorch, on nova.** The API image has no PyTorch (about 250 MB instead of several GB) and inference is faster on CPU. The export stage proves ONNX and PyTorch outputs match before the model can be registered.

**MLflow for model packaging and the API-fication path.** Every model is logged with `mlflow.onnx.log_model` (with a signature and input example), so it can be served by MLflow's own REST server unchanged: the optional `mlflow-serve` compose profile runs `mlflow models serve` on the champion. The main API uses the same registry artifact but adds what MLflow serving lacks: image decoding, the camera loop, drift monitoring and custom metrics.

**MLproject for identical environments.** `MLproject` + `python_env.yaml` define one environment. `mlflow run . -e test` and `mlflow run . -e main` build it and run the tests and the pipeline in it, so development and test use the same packages.

**Deploy by alias.** Serving loads `models:/roomsense@champion`. Deploying = moving the alias; rolling back = moving it back. The API polls the alias every 60 s and swaps the model in memory under a lock, so there is no restart and no dropped request.

**Promotion gate.** `register.py` promotes a candidate only if its macro-F1 is at least the champion's, with both evaluated on the same frames: the candidate's test sessions that the champion never trained or validated on. Comparing stored numbers from different splits would be meaningless, and so would scoring both on the candidate's whole test split: the splitter reassigns sessions when the dataset grows, so part of it can be the champion's old training data. The first real run on the doubled dataset showed exactly that: 62% of the test frames were sessions the champion had trained on, which put it at 0.73 against the candidate's 0.65. Each registered model therefore stores its `splits.csv` in its MLflow run, and the gate refuses to decide (no promotion) when fewer than 50 test frames are unseen by the champion.

**Rollback.** The replaced version is written into the new champion's tags. `scripts/rollback.py`, `POST /pipeline/rollback` and a button on the Pipeline page move the alias back; the API reloads immediately.

**Frontend and backend are separate, loosely coupled services.** Two Dockerfiles, two compose services. The frontend is static files behind nginx and knows the backend only through two environment variables: `API_UPSTREAM` (where nginx proxies `/api/`) and `API_BASE_URL` (what the browser calls). Either side can be replaced or moved without touching the other; a test fails if a page hard-codes an API address.

**Docker Compose, not Kubernetes.** One host, a dozen containers. Compose gives separate services, a private network, restart policies and health checks. Kubernetes would spend nova's 8 GB on itself.

### 4.5 Monitoring, drift and retraining

**Everything is scraped.** The API and the capture service export their own metrics; node-exporter covers the host, cAdvisor every container, nginx-exporter the frontend; Prometheus, Alertmanager and Grafana expose theirs. MLflow has no metrics page, so the API exports whether it can reach it. The pipeline's status and throughput reach Prometheus through the API too.

**Drift signal: share of frames in brightness bins the training data almost never had.** Lighting is the main thing that changes in this room. The Population Stability Index (PSI) is the textbook choice and is computed and displayed, but on a 120-frame window it is noisy and high even without drift, so alerting on it gave false alarms. The out-of-distribution share is near 0 in normal operation and near 1 when the light changes, which makes a 50% threshold for 5 minutes robust.

**A second trigger: live accuracy.** While a recording session runs, its label is ground truth for the live predictions. Accuracy under 60% over 30 labelled frames also requests a retrain. This catches drift that brightness cannot see.

**Pull-based retraining with a persistent flag.** Alertmanager calls the API; the API stores a flag on disk; the Legion polls it. The flag survives restarts, and nothing breaks if the Legion is off: retraining simply starts when it next checks.

**Recovery is observable.** The new model's reference histogram includes the new lighting, so the drift share drops and the alert resolves by itself.

## 5. Trade-offs accepted

| Choice | Cost | Why it is acceptable |
|---|---|---|
| SQLite for MLflow | One writer at a time | One developer, one training machine |
| Polling for model changes | Up to 60 s before a new champion serves | `/reload` and the retrain acknowledgement trigger it immediately |
| Retrain waits for the Legion | Not instant | The old model keeps serving; the alert tells the owner |
| Brightness-only drift alert | Blind to other shifts (furniture moved) | Live-accuracy and low-confidence alerts cover part of that |
| The split is recomputed when the data grows | Old test sessions can move into training, so the fair comparison set shrinks | The gate excludes them; a "sticky" split that keeps earlier assignments is the full fix |
| One camera angle | The model is useless elsewhere | Stated non-goal |

## 6. Security and privacy

- **Network.** `ufw` denies incoming traffic; Docker publishes only on 127.0.0.1; Tailscale Serve exposes HTTPS inside the tailnet; Tailscale Funnel (public exposure) is off. A test checks that no compose port is bound beyond localhost.
- **Encryption in transit.** All traffic between devices is WireGuard-encrypted by Tailscale, with HTTPS on top. At rest, the data sits on drives in a locked room; disk encryption is a possible addition.
- **Secrets.** Camera password, Telegram token and Grafana password live in `deploy/.env` (mode 600, git-ignored). State-changing endpoints accept a shared token (`RETRAIN_TOKEN`).
- **Data minimisation.** Frames are stored only during sessions the owner starts. The live view is not recorded. GitHub holds no images.
- **Consent.** The room has one occupant, who is the owner and operator.

## 7. Explainability

The project gives explanations at the level a user and an operator need: per-class probabilities for every prediction, calibrated confidence, a confusion matrix and per-class F1 for every training run in MLflow, k-means cluster maps showing which frames were kept, and a drift histogram comparing live and training brightness. Pixel-level attribution (Grad-CAM, a heat map of what the network looked at) was considered and left as future work: with a fixed camera the confusion matrix already shows which states are confused (`sitting_on_bed` vs `lying_in_bed`), and the fix for that is more varied data.

## 8. What would change at larger scale

Object storage (MinIO) as the DVC remote, PostgreSQL behind MLflow, a message queue instead of polling, a Ray cluster instead of one node, per-user authentication on the UI, and a canary step (serve the candidate to a share of frames) before moving the alias.
