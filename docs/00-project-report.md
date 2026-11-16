# RoomSense: project report

<!-- pdf:skip-start -->
**Khaja Mohammed** · DA3408 AIOps · IIT Madras · November 2026
<!-- pdf:skip-end -->

<!-- pdf:abstract-start -->
**Abstract.** RoomSense classifies the state of a hostel room into six classes from a fixed phone camera sampled every 5 s, and wraps the classifier in a complete, self-hosted machine learning lifecycle: versioned data collection, a reproducible DVC pipeline with Ray preprocessing, MLflow tracking and model registry, ONNX serving, Prometheus and Grafana monitoring, drift detection, alert-driven retraining and one-click rollback. Over 24 days, 56,796 labelled frames were collected in 108 sessions. The final model, a fine-tuned MobileNetV3-Small, reaches 0.518 test macro-F1 and was not promoted over the incumbent. The main finding concerns evaluation rather than modelling: re-splitting a growing dataset placed 62% of a candidate's test frames in the champion's training data, so the promotion gate was redesigned to compare models only on sessions unseen by both.
<!-- pdf:abstract-end -->

Companion documents: [architecture](01-architecture.md), [high-level design](02-hld.md), [low-level design](03-lld.md), [test plan](04-test-plan.md), [test report](05-test-report.md), [user manual](06-user-manual.md). Source code: `github.com/raisserv2/roomsense`.

---

## 1. Introduction

RoomSense infers the state of a hostel room from one fixed phone camera, one frame every 5 s. It distinguishes six classes: `at_desk`, `lying_in_bed`, `sitting_on_bed`, `in_room` (moving about), `empty` and `lights_off`.

The classifier is deliberately small; the subject of the project is the operational lifecycle around it. No cloud services are used, both for privacy and under the course rules. The system runs on two machines connected by a private Tailscale network, so every interface is reachable only from the owner's devices.

| Machine | Hardware | Role |
|---|---|---|
| **nova** | ThinkPad T460, 8 GB RAM, no GPU, always on | Data collection, model serving, MLflow, monitoring (Docker) |
| **Legion** | Laptop with an RTX 4060 GPU | Training pipeline |

## 2. System architecture

![System architecture: data collection and serving on nova, training on Legion, joined by Tailscale](diagrams/01-architecture.png)

The system operates as a closed loop:

1. **Collection.** A capture service on nova stores a labelled frame every 5 s during a recording session.
2. **Versioning.** Each night nova adds the new frames to DVC (Data Version Control), stored on a local drive, and pushes the dataset metadata to GitHub.
3. **Training.** On Legion, a six-stage DVC pipeline validates, preprocesses, splits, trains, exports to ONNX and registers the model in MLflow.
4. **Promotion and serving.** A candidate becomes `champion` only if it outperforms the incumbent on frames neither model has seen. The API loads the new champion without a restart.
5. **Monitoring and retraining.** Prometheus scrapes every component and Grafana displays them. A drift alert raises a retraining request; Legion picks it up and the loop restarts.

Each component is described in the [architecture document](01-architecture.md) and the rationale for each design choice in the [HLD](02-hld.md).

## 3. Dataset

**Collection.** Recording sessions are started and stopped from a phone page, so every frame is labelled at capture time and nothing is stored outside a session. A review tool allows frames to be relabelled or discarded; every change is logged.

| Property | Value |
|---|---|
| Frames | 56,796, collected 20 Oct to 12 Nov 2026 |
| Sessions | 108 |
| Lighting | 37,560 with the tube light on, 19,236 off |
| Time of day | 32,258 day, 24,538 night |

| Class | Frames | Sessions |
|---|---:|---:|
| empty | 27,170 | 31 |
| at_desk | 10,291 | 31 |
| lying_in_bed | 7,896 | 12 |
| sitting_on_bed | 5,367 | 20 |
| lights_off | 5,207 | 4 |
| in_room | 865 | 15 |

The classes are imbalanced, and the session count matters more than the frame count: consecutive frames are nearly identical, so `lights_off`, with four sessions, has far less variety than its 5,207 frames suggest.

**Validation.** The `prepare` stage checks every row: the file exists and is non-empty, the label and lighting values are known and compatible, the timestamp parses, and there are no duplicates. The pipeline fails if more than 5% of rows are invalid. No rows were dropped from the final dataset.

**Preprocessing.** The start and end of each session (walking to and from the phone) are trimmed, and frames are resized to 224 × 224. Within each session, k-means on image embeddings keeps the frame nearest each cluster centre (5 to 80 per session), reducing 56,202 frames to 2,419 so that long sessions do not dominate training.

**Splitting.** Whole sessions are assigned to train, validation and test (70/15/15), stratified by class, so no model is tested on a session it trained on. Final split: 1,696 / 371 / 352 frames from 54 / 26 / 28 sessions, with every class present in every split.

**Privacy.** Frames remain on the two machines and a backup drive; GitHub stores only dataset hashes. Automatic face blurring was evaluated and rejected because Haar detectors missed faces in profile and in low light (HLD, section 4.1).

## 4. Model and training

The classifier is MobileNetV3-Small pretrained on ImageNet, fine-tuned in two phases (classifier head with a frozen backbone, then the full network), with early stopping on validation macro-F1.

- **No flips or crops.** The desk is at the right of the frame and the bed at the left, so position carries the signal. Only colour jitter is applied.
- **Balanced sampling and label smoothing** to counter class imbalance and overconfidence.
- **Temperature scaling** fitted on the validation set, so that reported confidences are meaningful in the interface and in alerts.
- **ONNX export** with the temperature included. Registration requires the ONNX output to match PyTorch within 0.001; the final run's largest difference was 5.6 × 10⁻⁶.

Each run logs its parameters, per-epoch curves, per-class scores, confusion matrix and calibration plots to MLflow, tagged with the Git commit and DVC data version.

## 5. Results

### 5.1 Model versions

| Version | Date | Data | Test macro-F1 | Outcome |
|---|---|---|---:|---|
| v1 | 28 Oct | 5 classes, 857 training frames | 0.615 | First champion; biased toward `at_desk` and 99% confident when wrong |
| **v2** | early Nov | 6 classes, ~27k raw frames; balanced sampling, label smoothing, calibration | 0.548 | **Current champion.** No `lights_off` session in its test split; over the five classes present, about 0.66 |
| v3 | 12 Nov | ~57k raw frames | 0.652 | Not promoted: 0.652 vs 0.731 on a shared split, later shown to be unfair (section 6) |
| v4 | 12 Nov | v3 re-registered under the fair rule | (0.652) | Not promoted: 0.488 vs 0.519 on 129 frames unseen by both |
| v5 | 14 Nov | 56,796 raw frames (final run) | 0.518 | Not promoted: 0.432 vs 0.547 on 129 frames unseen by both |

Each score is measured on that version's own test split. Because the split changes as the dataset grows, scores in different rows are not directly comparable; the comparisons in the last column are.

### 5.2 Final model (v5)

| Class | F1 | Test frames |
|---|---:|---:|
| sitting_on_bed | 0.81 | 43 |
| at_desk | 0.79 | 142 |
| lights_off | 0.73 | 7 |
| in_room | 0.32 | 25 |
| empty | 0.25 | 90 |
| lying_in_bed | 0.25 | 45 |
| **Macro average** | **0.52** | 352 |

![Confusion matrix of v5 on its 352-frame test split](img/report-confusion-v5.png)

Two systematic errors dominate:

- **`empty` is predicted as `at_desk`** in 59 of 90 cases. With no person present the desk region dominates the frame, and `at_desk` is the most frequent occupied class. The same bias explains the near-perfect `at_desk` recall (141 of 142).
- **`lying_in_bed` is predicted as `in_room`** in 28 of 45 cases. A person under a blanket changes the outline of the bed only slightly when seen from above.

Both errors point to limited data variety rather than model capacity: more varied `empty` and `lying_in_bed` sessions (clothing, bedding, desk clutter, time of day) are likely to help more than a larger network.

**Calibration.** The temperature fitted on 371 validation frames (T = 1.84) made v5 underconfident on test: expected calibration error (ECE) rose from 0.081 to 0.160. With a validation set this small, temperature scaling should be cross-validated, or kept only when it lowers validation ECE.

### 5.3 Pipeline performance

![Duration of each pipeline stage in the final run](img/report-stage-times.png)

| Measure | Final run |
|---|---|
| End-to-end pipeline, 56,796 frames | 121 s |
| Ray resize | 6,573 frames/s (24 CPU cores) |
| Ray k-means de-duplication | 5,541 frames/s |
| Preprocessing, end to end | 1,445 frames/s |
| Training throughput | 1,248 images/s (RTX 4060) |
| Inference latency on nova (2 CPU threads) | 1.35 ms per frame (p95) |
| ONNX model size | 6.1 MB |

DVC reruns only stages whose inputs changed; a run with no new data completes in a few seconds from cache.

## 6. Fair model promotion

The original gate compared a candidate with the champion on the candidate's test split. After the dataset doubled, v3 lost to v2 by a wide margin (0.652 vs 0.731), which was implausible for a model trained on twice the data.

The splitter reassigns sessions whenever the dataset grows. Of v3's 338 test frames, **209 (14 of 22 sessions) came from sessions in v2's training set**, so v2 was largely being evaluated on data it had trained on.

The revised gate (`registry.fair_comparison_set` and the register stage) works as follows:

- Every registered model stores its split in MLflow.
- A candidate is compared with the champion **only on test sessions the champion never trained or validated on**, which are therefore unseen by both.
- If fewer than 50 such frames exist, promotion is declined with an explicit reason.

Under the fair gate, v2's score on the same candidate data fell from 0.731 to 0.519. Neither v3 (0.488) nor v5 (0.432 vs 0.547) surpassed it, so v2 remains the champion. With 129 frames, mostly one session per class, these differences are within noise: the gate is correct, but the shared test set is too small to separate the models. A split that never reassigns existing sessions would address this (section 9).

## 7. Deployment and operations

**Serving.** A FastAPI service on nova loads `models:/roomsense@champion` with ONNX Runtime, classifies the current frame every 5 s and polls the registry every 60 s. Deployment consists of moving the `champion` alias, and rollback of moving it back, which the Pipeline page exposes as a single action. The web frontend runs in a separate container and reaches the API only over REST at a configurable address. The API is specified in the [LLD](03-lld.md).

**Monitoring.** Prometheus scrapes eight jobs every 15 s: the API, the capture service, the frontend, the host, all containers, and Prometheus, Alertmanager and Grafana themselves. A 42-panel Grafana dashboard covers the model, the pipeline, data collection and component health. Thirteen alert rules notify the owner through Telegram.

**Drift detection.** The API compares the brightness of the last 120 frames with the training distribution. The alert signal is the share of recent frames in brightness ranges the training data rarely contains: close to 0 normally and close to 1 after a lighting change. The Population Stability Index is also reported but was too noisy on a 120-frame window to alert on. A second check compares live predictions with the label of an active recording session.

**Retraining.** A drift or accuracy alert calls the API's webhook, which persists a retraining request. Legion polls for it, pulls the latest data, reruns the pipeline, publishes the results and acknowledges the request. The demo video shows a request raised by the live-accuracy alert while Legion was offline and served once it came back.

**Pipeline console.** The Pipeline page shows the DVC graph with each stage's status, duration and throughput, data collection totals, the history of successful and failed runs, registered model versions and an activity log, with actions to request retraining, reload the model and roll back.

## 8. Testing

| Area | Result |
|---|---|
| Automated tests | 91 passing (unit, API, configuration) |
| Code style | `ruff` (PEP 8): no violations |
| Pipeline integration | Fresh run, fully cached run and forced failure, each recorded correctly |
| Acceptance criteria | 13 of 14 met |

The unmet criterion is AC-3, test macro-F1 of the latest model ≥ 0.60; v5 reached 0.518. The threshold was fixed in advance and not lowered after the result; sections 5 and 6 explain the shortfall. Every test case is listed in the test plan and every result in the generated test report.

## 9. Limitations and future work

- **Stable split.** Keep each session in the split it was first assigned to, so that test sessions stay unseen by all later models and comparisons become more reliable as data accumulates.
- **More varied sessions** for `lights_off` (four sessions), `lying_in_bed` and `empty`, the classes that limit macro-F1.
- **Conditional calibration.** Keep the fitted temperature only if it lowers validation ECE.
- **Explainability.** Grad-CAM saliency maps would show directly whether `empty` is being judged from the desk region.
- **Richer drift signals.** Distances in image-embedding space would detect changes that brightness cannot, such as moved furniture.
- **Scaling.** PostgreSQL behind MLflow, object storage for DVC and a canary stage before promotion for a multi-user deployment.

## 10. Conclusion

RoomSense runs the full lifecycle of a vision model on two personal machines with no cloud services: 56,796 frames collected and versioned, a 121-second reproducible pipeline, gated promotion, live monitoring with 13 alerts, and drift-triggered retraining. Model quality is limited by session variety rather than architecture. The most consequential result is methodological: when a dataset grows, the comparison between models must be restricted to data unseen by both, or the promotion gate rewards memorisation.

## Use of AI

Claude (Anthropic) was used as a coding assistant for drafting code, tests, documentation and diagrams. The project idea, design decisions, hardware and network setup, data collection and labelling, training and deployment runs, and the verification of every result are my own work.
