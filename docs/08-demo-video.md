# RoomSense: demo video shot list

Silent documentary style: no talking to camera, on-screen captions only. Target length 5 to 6 minutes. Screen recordings at 1080p; room shots on the phone, landscape, steady.

**Before recording**

- Final model trained and promoted; Pipeline page shows at least one ✓ run and one ✗ run (TC-INT-3 gives you the ✗).
- Telegram alerts working (`deploy/.env` filled, Alertmanager restarted).
- Browser at 100% zoom, bookmarks bar hidden, one tab per tool.
- For the drift scene, do the long waits for real once and speed them up in the edit (show a clock or keep the Grafana time axis visible so the speed-up is honest).

## Shots

| # | Length | Picture | Caption |
|---|---|---|---|
| 1 | 10 s | The room, wide. Then the phone on its shelf. | RoomSense · what is happening in this room? |
| 2 | 10 s | Architecture diagram (`docs/diagrams/01-architecture.png`), slow zoom | Two machines, no cloud. nova serves and monitors. The Legion trains. |
| 3 | 20 s | Phone: capture page. Tap **at_desk**, sit down, counter climbing, tap Stop. | 1 · Collect: labelled frames, one every 5 s, only while I record |
| 4 | 15 s | Review tool: scroll a session, trash two bad frames | Review: delete or relabel bad frames |
| 5 | 15 s | Terminal on nova: `systemctl list-timers roomsense-snapshot`, then `git log --oneline -3` showing snapshot commits | 2 · Version: nightly `dvc push` to my own drive; Git keeps the receipt |
| 6 | 25 s | Legion terminal: `dvc dag`, then `python scripts/run_pipeline.py` (speed up), stage lines scrolling, final "pipeline finished" | 3 · Pipeline: DVC DAG, six stages, only changed stages rerun |
| 7 | 10 s | `cat reports/preprocess_report.json` with `throughput_fps` highlighted; one cluster plot | Ray preprocessing: N frames/s · near-duplicates removed per session |
| 8 | 25 s | MLflow: runs list → one run → params, metric curves, confusion matrix, Tags (`git_commit`, `data_version`) | 4 · Track: every run reproducible from a Git commit + a run ID |
| 9 | 10 s | MLflow → Models → `roomsense`: versions, `champion` alias | 5 · Register: promoted only if better on the same test split |
| 10 | 30 s | Web UI Live page on the laptop. Walk from desk to bed in a picture-in-picture room shot; the label follows. | 6 · Serve: FastAPI + ONNX Runtime on a CPU, a few ms per frame |
| 11 | 10 s | Same page on the phone | Works on a phone |
| 12 | 15 s | "Try your own photo": upload a photo; then drop a PDF to show the friendly error | Any photo · clear messages when something is wrong |
| 13 | 30 s | Pipeline page: hover the DAG, select **preprocess**, scroll to Runs (✓ and ✗ rows), Model versions, Activity | 7 · Pipeline console: status, speed, failures, versions in one place |
| 14 | 20 s | Grafana dashboard, scrolling: model row, pipeline row, ingestion row, component status row | 8 · Monitor: model, data, pipeline and every container |
| 15 | 10 s | Prometheus → Targets (all UP) → Alerts (13 rules) | Every component scraped · 13 alert rules |
| 16 | 15 s | Caption card, then Live page showing the tube-only model (drift share ≈ 0%) | 9 · Drift demo. This model has only ever seen the tube light on. |
| 17 | 10 s | Room: hand switches the tube light off | Lights change |
| 18 | 25 s | Live page: drift share climbing past 50%, message turns red (speed up). Grafana drift panel crossing the threshold line. | Unseen lighting: 0% → 80% |
| 19 | 10 s | Phone: Telegram alert arrives. Pipeline page Activity: "Retrain requested". | Alert after 5 minutes → retrain flag raised |
| 20 | 25 s | Legion: `python scripts/retrain.py` (speed up): git pull, dvc pull, stages, "retrain done" | 10 · Retrain: the Legion picks up the flag and reruns the pipeline |
| 21 | 20 s | Live page: pill changes to the new version; drift share falls; Telegram "resolved". Pipeline page: new ✓ row, new version "in use". | 11 · Recover: new champion hot-reloaded, drift clears |
| 22 | 15 s | Pipeline page: **Go back to the previous model** → confirm → version changes; then switch forward again | Rollback in one click |
| 23 | 10 s | Terminal: `python scripts/test_report.py` ending in "86 passed"; the acceptance table in the report | 86 tests · acceptance criteria met |
| 24 | 8 s | Architecture diagram again, then title card | RoomSense · AIOps course project · Khaja Mohammed |

## Drift scene, exact commands

On the Legion, the day before (so the tube-only model is the champion when you record):

```bash
# 1. train a model that has only seen the tube light on
sed -i 's/^  include: .*/  include: {lighting: [tube_on]}/' params.yaml
sed -i 's/^  force: false/  force: true/' params.yaml          # promote it even though it scores lower
python scripts/run_pipeline.py --trigger demo --reason "tube-only model for the drift demo"
```

Record shots 16 to 19 (tube off, wait for the alert). Then:

```bash
# 2. give the pipeline all the data again and let the retrain loop run
sed -i 's/^  include: .*/  include: {}/' params.yaml
sed -i 's/^  force: true/  force: false/' params.yaml
python scripts/retrain.py            # sees the flag, reruns, promotes, acknowledges
```

Check `params.yaml` after each `sed` (the `data.include` and `register.force` lines) before running.

## Edit notes

- Captions bottom-left, one font, white on a dark translucent bar, on screen at least 3 s.
- Speed-ups 8× to 20× with a small "×N" tag.
- No music needed; if any, low and without vocals.
- Blur anything personal in room shots.
- Export 1080p H.264.
