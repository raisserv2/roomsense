# RoomSense: viva question sheet

Short answers to say out loud, grouped by topic. Each ends with where the evidence is, so you can open it if asked.

Note: section 9 covers the standard MLOps (Machine Learning Operations) tool topics. Check it against your own Assignments 1–10 and add anything specific they asked.

## 1. The 60-second explanation

"RoomSense tells what state my hostel room is in from a phone camera: at desk, in bed, empty, lights off and so on. The model is small, a MobileNetV3. The project is the loop around it. A capture service on my server, nova, records labelled frames. A DVC pipeline on my GPU laptop checks the data, preprocesses it in parallel with Ray, splits by session, trains, exports to ONNX and registers the model in MLflow, promoting it only if it beats the current one. nova serves the champion with FastAPI and ONNX Runtime, a separate nginx frontend shows it, and Prometheus and Grafana monitor every component. When the lighting changes, a drift alert fires, raises a retrain flag, the laptop reruns the pipeline, the API hot-reloads the new model and the alert clears. No cloud anywhere."

## 2. Data

**Why label during capture instead of afterwards?** Labelling 27 000 frames by hand later would not happen. One tap labels a whole session. It also means nothing is recorded unless I start a session, which is my privacy rule.

**Why split by session?** Frames 5 seconds apart are near-copies. A random split would put copies in train and test and the score would be inflated (leakage). I assign whole sessions and a test proves no session is in two splits. *(`scripts/split.py`, `tests/test_split.py`)*

**Why throw frames away with k-means?** A 10-minute session is 120 near-identical pictures. I embed each frame, cluster per session and keep the frame nearest each centre, 5 to 80 per session. It keeps variety, removes duplicates, and stops long sessions dominating. Taking every n-th frame would keep duplicates of a still scene.

**What are your data quality checks?** In `prepare`: file exists and is not empty, label and lighting are known, the pair is possible, the timestamp parses, no duplicate paths. Each failure is counted by reason, and over 5% bad rows fails the pipeline. *(`reports/prepare_report.json`)*

**Why no flips or crops?** The desk is on the right edge and the bed on the left. Position is the signal. A flip would turn "at desk" into nonsense. Only colour jitter.

**Your classes are imbalanced. What did you do?** A weighted sampler balanced by class and by session, label smoothing, macro-F1 instead of accuracy, and collection targets per class.

**Why did you switch face blurring off?** Haar cascades missed profile and low-light faces and blurred random patches. A privacy filter that works sometimes is worse than none because you trust it. Privacy is handled by keeping images on my two machines.

## 3. Pipeline, versioning, CI

**What does DVC give you that Git doesn't?** Git cannot hold gigabytes of images. DVC stores them on my Seagate by content hash and puts a small pointer file in Git. So one Git commit pins code, data and parameters together.

**How is the DVC DAG "continuous integration"?** `dvc repro` rebuilds only the stages whose inputs or params changed and stops at the first failure. That is what a CI server does: on change, rebuild and validate. I can't use a cloud CI, so the DAG is the local one, and `run_pipeline.py` records every run with pass/fail.

**How do I reproduce one of your experiments?** The MLflow run has tags `git_commit` and `data_version` and stores `dvc.lock`. `git checkout <commit>`, `dvc pull`, `dvc repro` gives the same model. *(Open any run in MLflow → Tags.)*

**Why Ray and not Spark?** One machine, embarrassingly parallel image work. Ray is a pip install and a decorator; Spark needs a JVM and brings a cluster model I don't need. The same Ray code would scale to more nodes.

**What is the throughput?** The preprocess report has seconds and frames per second for each step (Ray resize, embedding, Ray k-means, end to end). The Pipeline page and Grafana show them. *(Read the current number off the Pipeline page before the viva.)*

**What happens when a stage fails?** The stage's exception hook writes the error to `reports/stages/<stage>.json`, DVC stops, the wrapper records a FAILED run in MLflow with the failed stage and log tail. The Pipeline page shows a red row and a Prometheus alert fires.

## 4. Model and training

**Why MobileNetV3-Small?** About 900 training frames after de-duplication and a CPU-only server. It has around 1.5 million parameters, the ONNX file is 6 MB, and a prediction takes a few milliseconds. A bigger model would overfit and gain nothing.

**What is temperature calibration and why?** My first model was 99% confident when wrong. I fit one number T on the validation set and divide the logits by it. Accuracy doesn't change; confidence becomes honest, so the "how sure" bar and the low-confidence alert mean something. I log ECE (Expected Calibration Error) before and after.

**Why macro-F1 and not accuracy?** Accuracy rewards getting the big class right. Macro-F1 averages per-class F1 so every class counts equally.

**Your macro-F1 ignores absent classes. Isn't that cheating?** If a class has zero test frames, its F1 is undefined; counting it as 0 measures my split, not my model. I average over classes that are present and I report which were absent, and my acceptance criteria require that none are absent in the final run. Champion and candidate are always scored the same way on the same frames.

**How do you decide a new model is better?** `register.py` downloads the champion and evaluates both models on the candidate's test sessions that the champion never trained on. Promote only if candidate ≥ champion there. Comparing stored numbers from different splits would be meaningless.

**Why "sessions the champion never trained on"?** I found this the hard way. My splitter reassigns sessions when the dataset grows. When the data doubled, 62% of the new test frames were sessions v2 had trained on, so v2 scored 0.73 against the new model's 0.65: a memory test, not a fair one. Now every registered model stores its split in MLflow, the gate compares only on sessions unseen by both, and it refuses to decide on fewer than 50 such frames. The full fix would be a sticky split that never moves old sessions.

**What is your weakest class and why?** Sitting on the bed versus lying in bed: same place, similar pose from above. And `lights_off` had very few sessions. The fix is more varied sessions, which is a data problem, not a model problem.

## 5. Experiment tracking

**What do you log beyond autolog?** I don't use autolog. I log manually: every hyper-parameter, per-epoch loss and validation F1, test metrics per class, the temperature, ECE before/after, the confusion matrix image, the reports of earlier stages, `dvc.lock`, and tags for Git commit, data version and device. Pipeline runs are a second experiment with per-stage seconds, throughput and status.

**Tracking versus registry?** Tracking is the notebook: every run, good or bad. The registry is the shelf: versions of the one model I serve, with an alias saying which is live.

**Why an alias instead of stages?** MLflow deprecated the Staging/Production stages. An alias is just a movable pointer: `champion`.

## 6. Serving and packaging

**Why ONNX?** nova has no GPU. ONNX Runtime is faster on CPU and the image needs no PyTorch, hundreds of MB instead of gigabytes. Export checks that ONNX and PyTorch outputs match to 1e-3.

**Where do you use MLflow for API-fication?** The model is logged with `mlflow.onnx.log_model` with a signature, so `mlflow models serve` can serve it as REST unchanged; that is the `mlflow-serve` compose profile. My main API loads the same registry artifact but adds image decoding, the camera loop, drift and metrics, which MLflow serving doesn't do.

**What does MLproject do for you?** It names the entry points (main, train, export, register, retrain, rollback, test) and points to `python_env.yaml`. `mlflow run . -e test` builds that environment and runs the tests in it, so dev and test environments are the same.

**How is the frontend loosely coupled?** Separate container, static files. It knows the backend only through two environment variables: where nginx proxies `/api/`, and the base URL the browser calls. No page has an address in it; a test checks that. I could put the API on another machine by changing one line in `.env`.

**How does a new model go live without downtime?** The API polls the `champion` alias every 60 seconds. If it moved, it downloads the model, builds a new session, and swaps a reference under a lock. In-flight requests finish on the old one.

**How do you roll back?** Promotion writes the replaced version into a tag. Rollback moves the alias back: a button on the Pipeline page, `scripts/rollback.py`, or the endpoint. The API reloads at once. Nothing is deleted.

**Why Compose and not Kubernetes?** One 8 GB host. Compose gives me separate services, a network, restart policies and health checks. Kubernetes would use the memory I need for the services.

## 7. Monitoring and drift

**What do you monitor?** Model: latency, confidence, class mix, live accuracy. Data: brightness, drift share, PSI. Pipeline: last run status, stage durations, throughput. Ingestion: frames saved, camera errors, dataset size. Infrastructure: host CPU, memory, disk, temperature, every container, nginx requests, and whether each component is up. 13 alert rules.

**Are all components monitored?** Yes: API and capture export metrics; node-exporter, cAdvisor and nginx-exporter cover host, containers and frontend; Prometheus, Alertmanager and Grafana expose their own. MLflow has no metrics page, so the API exports whether it can reach it and cAdvisor covers its container.

**Why not alert on PSI?** PSI (Population Stability Index) on a 120-frame window is noisy; it was high with no drift. I alert on the share of frames falling in brightness bins the training data almost never had: near 0 normally, near 1 when the light changes. PSI is still shown.

**Is that data drift or concept drift?** Data (covariate) drift: the input distribution changed, the meaning of the labels didn't. I also have a live-accuracy alert, which would catch concept drift while a labelled session is running.

**What if drift isn't about brightness?** The brightness signal won't see it. Live accuracy and the low-confidence alert cover part of it. A better signal would be distance in embedding space; that's future work.

**Why does the Legion poll instead of nova pushing?** The Legion is often off. A flag on disk on nova survives restarts and nothing fails if nobody is listening. nova never needs credentials for the Legion.

**Could retraining loop forever?** The flag is raised once per firing alert and cleared on acknowledgement. If the new model isn't better it isn't promoted, the alert stays, and I get told. It needs new data to succeed, which is the honest answer to drift.

## 8. Engineering

**How did you test?** 91 automated tests: unit tests of the library, API tests with a tiny generated ONNX model and a fake MLflow, configuration tests. Plus pipeline integration runs, live checks and manual UI cases. Acceptance criteria are in `params.yaml` and checked by a script that writes the test report.

**How do you handle errors?** Bad input gets a 4xx with a readable message; missing dependencies a 503; anything unexpected is logged with an id and returns a clean 500. Background loops catch, count and continue. MLflow calls time out fast so a dead tracker can't hang the API.

**What about security?** Tailnet-only access, WireGuard encryption, HTTPS on top, all Docker ports on 127.0.0.1, firewall deny-incoming, secrets in a git-ignored `.env`, a token on state-changing endpoints.

**What would you do differently with more time?** Embedding-based drift, Grad-CAM for explanations, a canary step before promotion, PostgreSQL and MinIO instead of SQLite and a USB drive, more `lights_off` and bed sessions.

## 9. Tool quick reference

| Tool | One line | Where in RoomSense |
|---|---|---|
| **Git** | Versions text: code and pointers | Repository; `dvc.lock`, `data/raw.dvc` |
| **DVC** | Versions data and models by hash; runs a DAG of stages | `dvc.yaml`, remote on the Seagate |
| **Ray** | Parallel Python tasks on one or many machines | `preprocess.py`: `@ray.remote` resize and k-means |
| **MLflow Tracking** | Logs params, metrics, artifacts per run | Experiments `roomsense`, `roomsense-pipeline` |
| **MLflow Registry** | Versioned models with aliases | Model `roomsense`, alias `champion` |
| **MLflow Models / serve** | Standard packaging and a REST server | `mlflow.onnx.log_model`; `mlflow-serve` profile |
| **MLflow Projects** | Reproducible entry points + environment | `MLproject`, `python_env.yaml` |
| **ONNX / ONNX Runtime** | Portable model format and a fast CPU engine | `export.py`; the API |
| **FastAPI** | Python web framework with automatic OpenAPI docs | Inference API, capture service |
| **Docker / Compose** | Package each service; run them together | `deploy/compose.yaml`: 9 services + 1 optional |
| **Prometheus** | Pulls metrics, stores time series, evaluates alert rules | 8 scrape jobs, 13 rules |
| **Alertmanager** | Groups and routes alerts | Telegram + webhook to the API |
| **Grafana** | Dashboards on Prometheus data | 42 panels, 10 s refresh |
| **Exporters** | Turn something into `/metrics` | node-exporter, cAdvisor, nginx-exporter |
| **Tailscale** | Private encrypted network between my devices | The only way in |

**Terms to have ready:** data drift vs concept drift · leakage · calibration and ECE · macro vs micro F1 · precision/recall · transfer learning · champion/challenger · canary vs blue-green vs rollback · pull (Prometheus) vs push monitoring · counter vs gauge vs histogram · idempotent stage · artifact vs parameter vs metric · image vs container · bind mount vs volume.
