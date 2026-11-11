"""Draws every RoomSense diagram:  python docs/diagrams/make_diagrams.py [name ...]

Output: SVG + PNG + PDF next to this file. Needs `rsvg-convert` (apt install
librsvg2-bin) for PNG/PDF and, optionally, Pillow + the Inter font for exact
text widths.

Real pictures: drop `frame.jpg` (a camera frame), `ui-live.png` and
`ui-grafana.png` (screenshots) into ./assets and rerun; the figures pick them
up instead of the drawn sample frame.
"""
import sys

from figlib import ASSETS, CONTROL, MUTED, Fig, text_width


def asset(*names):
    """First existing picture among the given asset names (None when there is none)."""
    return next((ASSETS / n for n in names if (ASSETS / n).exists()), None)


FRAME = asset("frame.jpg", "frame.png", "sample-frame.jpg")


# 1 --------------------------------------------------------------------------
def scraped(f, bid):
    """Small pulse badge: this block exposes metrics that Prometheus scrapes."""
    x, y, w, _h = f.boxes[bid]
    f.icon(x + w - 20, y + 5, 13, "lu-activity", "#6A4BA8")


def architecture():
    f = Fig(1200, 760, "RoomSense system architecture")
    f.lane(16, 44, 196, 700, "Room", "lu-lightbulb")
    f.lane(226, 44, 620, 700, "nova", "lu-server", "ThinkPad T460 · Docker · capture, serving, monitoring")
    f.lane(872, 44, 312, 470, "Legion", "lu-cpu", "RTX 4060 · training")
    f.lane(872, 528, 312, 216, "Outside the room", "lu-globe-lock")

    # room
    f.box("cam", 30, 100, 168, 58, "Phone camera", ["Honor 7X · IP Webcam"], "ext", icon="lu-smartphone")
    if FRAME:
        f.image(48, 176, 132, 99, FRAME, caption="640×480 frame every 5 s")
    f.box("user", 30, 540, 168, 62, "You", ["browser on a phone", "or laptop"], "ext", icon="lu-user")

    c1, c2, c3, w = 244, 448, 652, 176
    # row 1: data
    f.box("capture", c1, 100, w, 62, "Capture service", ["labels · review tool"], "data", logo="fastapi")
    f.box("raw", c2, 100, w, 62, "Raw frames", ["data/raw + frames.csv"], "data", icon="lu-image")
    f.box("remote", c3, 100, w, 62, "DVC remote", ["Seagate drive"], "store", logo="dvc")
    # row 2: serving + registry
    f.box("api", c1, 236, w, 70, "Inference API", ["ONNX Runtime · drift", "camera loop"], "serve", logo="fastapi")
    f.box("mlflow", c3, 236, w, 70, "MLflow", ["tracking · registry", "champion alias"], "store", logo="mlflow")
    # row 3
    f.box("front", c1, 400, w, 62, "Web frontend", ["Live · Pipeline · Help"], "serve", logo="nginx")
    f.box("prom", c2, 400, w, 62, "Prometheus", ["scrapes · 13 alert rules"], "monitor", logo="prometheus")
    f.box("graf", c3, 400, w, 62, "Grafana", ["live dashboard"], "monitor", logo="grafana")
    # row 4
    f.box("serve", c1, 540, w, 62, "Tailscale Serve", ["HTTPS, tailnet only"], "ext", logo="tailscale")
    f.box("am", c2, 540, w, 62, "Alertmanager", ["routes alerts"], "monitor", icon="lu-bell")
    f.box("exp", c3, 540, w, 62, "Exporters", ["host · containers · nginx"], "monitor", icon="lu-gauge")
    for bid in ("capture", "api", "graf", "am", "exp"):
        scraped(f, bid)
    f.icon(c1, 712, 13, "lu-activity", "#6A4BA8")
    f.text(c1 + 18, 723, "= exposes /metrics, scraped by Prometheus every 15 s", 11, fill=MUTED)

    # Legion: the DVC pipeline
    px, pw, ph = 892, 272, 40
    stages = [("prepare", "data quality checks", "dvc"), ("preprocess", "trim · resize · de-duplicate", "ray"),
              ("split", "by session, no leakage", "dvc"), ("train", "MobileNetV3-Small, GPU", "pytorch"),
              ("export", "ONNX · parity · latency", "onnx"), ("register", "promote only if better", "mlflow")]
    for i, (name, sub, logo) in enumerate(stages):
        y = 96 + i * 52
        f.box(name, px, y, pw, ph, name, [], "train", logo=logo, title_size=13)
        f.text(px + 36 + text_width(name, 13, True) + 10, y + 25, sub, 11, fill=MUTED)
        if i:
            f.arrow([(px + 20, y - 12), (px + 20, y)], width=1.3)
    f.text(px, 86, "DVC DAG, run by scripts/run_pipeline.py", 11, fill=MUTED)
    f.box("retrain", px, 446, pw, 50, "retrain.py", ["polls the retrain flag, reruns the DAG"], "train", icon="lu-refresh-cw")
    # outside
    f.box("github", 892, 610, 130, 54, "GitHub", ["code · dvc.lock"], "ext", logo="github")
    f.box("tg", 1036, 610, 128, 54, "Telegram", ["alerts"], "ext", logo="telegram")

    # arrows: the loop
    f.arrow([f.anchor("cam", "r"), f.anchor("capture", "l", -2)], step=1, step_at=0.5)
    f.link("capture", "raw")
    f.link("raw", "remote", label="nightly dvc push", ldy=-42, label_bg=False, step=2, step_at=0.5)
    f.arrow([f.anchor("remote", "r", -10), (859, 121), (859, 116), f.anchor("prepare", "l")], label="dvc pull", lpos=0.45,
            ldy=19, step=3, step_at=0.45)
    f.arrow([f.anchor("register", "l"), (853, 376), (853, 271), f.anchor("mlflow", "r")], label="runs + model version",
            lpos=1.0, ldx=-52, ldy=50, step=4, step_at=0.55)
    f.link("mlflow", "api", "l", "r", shift_a=-14, shift_b=-14, label="champion (hot reload)", step=5, step_at=0.12)
    f.arrow([(180, 240), (222, 240), (222, 271), f.anchor("api", "l")] if FRAME else
            [f.anchor("cam", "b"), (114, 271), f.anchor("api", "l")], step=6, step_at=0.55)
    f.arrow([f.anchor("front", "t", -40), f.anchor("api", "b", -40)], label="REST /api", both=True, step=7, ldx=44,
            step_at=0.5)
    f.arrow([f.anchor("user", "r"), f.anchor("serve", "l")])
    f.arrow([f.anchor("serve", "t", -40), f.anchor("front", "b", -40)])
    f.arrow([f.anchor("api", "r", 6), (536, 277), f.anchor("prom", "t")], label="/metrics", step=8, lpos=0.5, ldx=34,
            step_at=0.5)
    f.link("prom", "graf", label="", width=1.6)
    f.arrow([f.anchor("prom", "b"), f.anchor("am", "t")], label="alerts", dashed=True, step=9, ldx=34, step_at=0.5)
    f.arrow([f.anchor("am", "l", -10), (434, 561), (434, 296), f.anchor("api", "r", 25)],
            label="webhook:\nretrain flag", dashed=True, step=10, lpos=0.22, step_at=0.5)
    f.arrow([f.anchor("am", "b"), (536, 690), (1100, 690), f.anchor("tg", "b")], label="alert message", dashed=True,
            lpos=0.45)
    f.arrow([f.anchor("retrain", "l"), (864, 471), (864, 356), (384, 356), f.anchor("api", "b", 52)],
            label="polls flag, acknowledges", dashed=True, step=11, lpos=0.5, step_at=0.3)
    f.arrow([f.anchor("register", "r"), (1174, 376), (1174, 580), (957, 580), f.anchor("github", "t")],
            label="git push", lpos=0.8, width=1.3)
    f.legend(28, 34, lines=True, steps=True)
    return f


# 2 --------------------------------------------------------------------------
def pipeline():
    f = Fig(1200, 548, "RoomSense data and training pipeline (DVC DAG)")
    f.lane(16, 44, 150, 250, "nova", "lu-server")
    f.lane(180, 44, 1004, 250, "Legion", "lu-cpu", "DVC DAG: one stage reruns only when its inputs or params changed")
    f.lane(16, 306, 1168, 226, "What is stored where", "lu-database")

    f.box("raw", 28, 96, 126, 62, "data/raw", ["frames.csv", "+ JPEGs"], "data", icon="lu-image")
    f.box("remote", 28, 218, 126, 58, "DVC remote", ["Seagate"], "store", logo="dvc")

    names = [("prepare", "prepare", ["data quality", "checks, day/night"], "dvc"),
             ("preprocess", "preprocess", ["Ray: trim, resize,", "k-means de-dup"], "ray"),
             ("split", "split", ["whole sessions", "70 / 15 / 15"], "dvc"),
             ("train", "train", ["MobileNetV3-Small", "calibration"], "pytorch"),
             ("export", "export", ["ONNX, parity,", "CPU latency"], "onnx"),
             ("register", "register", ["champion only", "if better"], "mlflow")]
    x0, bw, gap, y = 198, 146, 20, 96
    arts = [["manifest.csv", "prepare_report.json"], ["images/ (224×224)", "clean_manifest.csv", "preprocess_report.json"],
            ["splits.csv", "split_report.json"], ["best.pt", "calibration.json", "metrics.json"],
            ["model.onnx", "model_meta.json", "export_report.json"], ["registry version", "register_report.json"]]
    for i, (bid, title, sub, logo) in enumerate(names):
        x = x0 + i * (bw + gap)
        f.box(bid, x, y, bw, 70, title, sub, "train", logo=logo)
        f.step(x + 6, y + 2, i + 1)
        if i:
            f.link(names[i - 1][0], bid)
        cy = y + 86
        for a in arts[i]:
            role = "data" if a.endswith((".csv", "/ (224×224)")) else "store"
            f.chip(x, cy, a, "train" if a.endswith("_report.json") or a == "metrics.json" else role, size=9.8)
            cy += 24
        f.arrow([(x + bw / 2, y + 70), (x + bw / 2, y + 84)], width=1.1)
    f.arrow([f.anchor("raw", "r"), f.anchor("prepare", "l", -4)], label="")
    f.arrow([f.anchor("raw", "b"), f.anchor("remote", "t")], label="nightly\ndvc push", width=1.3)
    f.arrow([f.anchor("remote", "r"), (176, 247), (176, 150), f.anchor("prepare", "l", 20)], label="dvc pull", lpos=0.42,
            ldx=0, width=1.3)
    f.text(198, 86, "scripts/run_pipeline.py runs  dvc repro  and records the run: status, seconds and throughput of every stage",
           11, fill=MUTED)

    # stores
    sx, sw = 34, 270
    stores = [("git", "Git + GitHub", "github", "ext", ["code, dvc.yaml, params.yaml", "dvc.lock (exact hashes)", "data/raw.dvc (data receipt)",
                                                         "reports/*.json, reports/stages/"]),
              ("dvcs", "DVC cache + remote", "dvc", "store", ["data/raw (every snapshot)", "data/processed/images", "models/best.pt, model.onnx",
                                                              "pulled / pushed by hash"]),
              ("mlf", "MLflow tracking (nova)", "mlflow", "store", ["params, per-epoch metrics", "confusion matrix, reports", "tags: Git commit, data version",
                                                                    "pipeline runs: status, timings"]),
              ("reg", "MLflow model registry", "mlflow", "store", ["ONNX model + metadata", "version tags: F1, decision", "alias  champion  = deployed",
                                                                   "previous champion = rollback"])]
    for i, (bid, title, logo, role, lines) in enumerate(stores):
        x = sx + i * (sw + 18)
        f.box(bid, x, 344, sw, 44, title, [], role, logo=logo)
        for j, line in enumerate(lines):
            f.add("node", f'<circle cx="{x + 14}" cy="{410 + j * 19 - 4}" r="2.2" fill="{MUTED}"/>')
            f.text(x + 24, 410 + j * 19, line, 11.5, fill="#374151")
    f.text(34, 514, "Reproduce any experiment:  git checkout <commit>  →  dvc pull  →  dvc repro   (the MLflow run stores the commit and the data version)",
           11.5, fill="#374151", bold=True)
    f.arrow([f.anchor("train", "t", 30), (f.boxes["train"][0] + 103, 70), (1100, 70)], label="logs run", dashed=True, lpos=0.6)
    f.text(1106, 74, "MLflow", 11.5, bold=True, fill=CONTROL)
    f.legend(28, 34, roles=("data", "train", "store", "ext"), lines=True, steps=True)
    return f


# 3 --------------------------------------------------------------------------
def serving():
    f = Fig(1200, 524, "RoomSense serving and monitoring")
    f.lane(16, 44, 180, 464, "Room", "lu-lightbulb")
    f.lane(210, 44, 974, 464, "nova", "lu-server", "Docker Compose, network roomsense")
    f.box("cam", 28, 96, 156, 58, "Phone camera", ["IP Webcam"], "ext", icon="lu-smartphone")
    if FRAME:
        f.image(48, 170, 116, 87, FRAME)
    f.box("user", 28, 420, 156, 58, "You", ["browser, tailnet"], "ext", icon="lu-user")

    # serving row
    f.box("capture", 232, 300, 180, 58, "Capture service", ["active session label"], "data", logo="fastapi")
    scraped(f, "capture")
    f.add("node", '<rect x="446" y="84" width="300" height="196" rx="12" fill="#F2FAF5" stroke="#62B283" stroke-width="1.4"/>')
    f.boxes["api"] = (446, 84, 300, 196)
    f.icon(458, 93, 20, "fastapi")
    f.text(484, 108, "Inference API", 13.5, bold=True)
    f.text(590, 108, "FastAPI, one process", 11, fill=MUTED)
    f.box("loop", 460, 122, 130, 44, "Camera loop", ["every 5 s"], "serve", title_size=12)
    f.box("model", 604, 122, 130, 44, "Model", ["ONNX Runtime"], "serve", logo="onnx", title_size=12)
    f.box("drift", 460, 180, 130, 44, "Drift window", ["last 120 frames"], "serve", title_size=12)
    f.box("mgr", 604, 180, 130, 44, "Model manager", ["polls alias, 60 s"], "serve", title_size=12)
    f.text(460, 250, "/predict /latest /drift /pipeline", 10.5, fill=MUTED, mono=True)
    f.text(460, 266, "/retrain /events /metrics", 10.5, fill=MUTED, mono=True)
    f.box("mlflow", 830, 96, 170, 62, "MLflow registry", ["roomsense@champion"], "store", logo="mlflow")
    f.box("front", 446, 420, 300, 58, "Web frontend (nginx)", ["Live · Pipeline · Help   →   /api/ proxy"], "serve", logo="nginx")
    f.box("serve", 232, 420, 180, 58, "Tailscale Serve", ["HTTPS :4443"], "ext", logo="tailscale")

    # monitoring column
    f.box("prom", 830, 206, 170, 62, "Prometheus", ["scrape every 15 s"], "monitor", logo="prometheus")
    f.box("graf", 1024, 206, 146, 62, "Grafana", ["42 panels"], "monitor", logo="grafana")
    f.box("am", 830, 320, 170, 58, "Alertmanager", ["13 alert rules"], "monitor", icon="lu-bell")
    f.box("tg", 1024, 320, 146, 58, "Telegram", ["to your phone"], "ext", logo="telegram")
    f.box("exp", 830, 420, 340, 58, "Exporters", ["node-exporter · cAdvisor · nginx-exporter"], "monitor",
          icon="lu-gauge")

    f.arrow([(164, 213), (200, 213), (200, 144), f.anchor("loop", "l")] if FRAME else [f.anchor("cam", "r"), f.anchor("loop", "l")],
            label="frame", step=1, lpos=0.8, ldy=-10, step_at=0.55)
    f.arrow([f.anchor("loop", "r"), f.anchor("model", "l")], width=1.3)
    f.arrow([f.anchor("loop", "b"), f.anchor("drift", "t")], width=1.3)
    f.arrow([f.anchor("mgr", "t"), f.anchor("model", "b")], width=1.3)
    f.arrow([f.anchor("capture", "t"), (322, 202), f.anchor("drift", "l")], label="truth label", lpos=0.45, width=1.3)
    f.arrow([f.anchor("mlflow", "l"), (788, 127), (788, 202), f.anchor("mgr", "r")], label="champion", step=2, lpos=0.42,
            step_at=0.12)
    f.arrow([f.anchor("front", "t"), f.anchor("api", "b")], label="REST (configurable URL)", both=True, step=3, ldx=84,
            step_at=0.5)
    f.arrow([f.anchor("user", "r"), f.anchor("serve", "l")], label="HTTPS")
    f.arrow([f.anchor("serve", "r"), f.anchor("front", "l")])
    f.arrow([(746, 237), f.anchor("prom", "l")], label="/metrics", step=4, step_at=0.2, lpos=0.6, ldy=-11, label_bg=False)
    f.arrow([f.anchor("exp", "t", -130), (870, 400), (812, 400), (812, 252), f.anchor("prom", "l", 15)], width=1.2)
    f.link("prom", "graf", step=5, step_at=0.5)
    f.arrow([f.anchor("prom", "b"), f.anchor("am", "t")], label="alerts", dashed=True, step=6, ldx=30, step_at=0.5)
    f.link("am", "tg", dashed=True)
    f.arrow([f.anchor("am", "l"), (776, 349), (776, 304), (700, 304), (700, 280)], label="webhook: retrain flag", dashed=True,
            step=7, lpos=0.5, ldy=13, step_at=0.12)
    f.legend(28, 34, roles=("data", "serve", "monitor", "store", "ext"), lines=True, steps=True)
    return f


# 4 --------------------------------------------------------------------------
def drift_sequence():
    f = Fig(1200, 712, "Drift, retrain and recovery as a sequence")
    def band(y, h, label, fill):
        f.add("lane", f'<rect x="12" y="{y}" width="1176" height="{h}" rx="10" fill="{fill}"/>')
        f.text(1180, y + 16, label, 11, bold=True, fill="#4B5563", anchor="end", layer="lane")

    band(108, 214, "DRIFT", "#FDF3F0")
    band(326, 178, "RETRAIN", "#FFF8EE")
    band(508, 190, "RECOVERY", "#F1F9F4")
    actors = [("room", "Room + camera", "lu-lightbulb", "ext", None), ("api", "Inference API", None, "serve", "fastapi"),
              ("prom", "Prometheus", None, "monitor", "prometheus"), ("am", "Alertmanager", "lu-bell", "monitor", None),
              ("you", "You (Telegram)", None, "ext", "telegram"), ("legion", "Legion: retrain.py", "lu-refresh-cw", "train", None),
              ("mlflow", "MLflow registry", None, "store", "mlflow")]
    xs = {}
    for i, (bid, title, icon, role, logo) in enumerate(actors):
        x = 24 + i * 167
        f.box(bid, x, 50, 150, 46, title, [], role, logo=logo, icon=icon, title_size=12)
        xs[bid] = x + 75
        f.add("lane", f'<path d="M{x + 75},96 V698" stroke="#B8C2CC" stroke-width="1.2" stroke-dasharray="3 5"/>')

    def msg(n, a, b, y, text, dashed=False, note=""):
        x1, x2 = xs[a], xs[b]
        if a == b:  # something the actor does itself
            f.add("node", f'<rect x="{x1 - 8}" y="{y - 13}" width="16" height="26" rx="4" fill="#fff" stroke="{CONTROL if dashed else "#3D4756"}" stroke-width="1.4"/>')
            f.step(x1 - 22, y, n, CONTROL if dashed else "#3D4756")
            f.text(x1 + 16, y + 4, text, 11.5, fill="#1F2933")
            if note:
                f.text(x1 + 16, y + 19, note, 10.5, fill=MUTED)
            return
        f.arrow([(x1, y), (x2, y)], dashed=dashed)
        left = min(x1, x2)
        f.step(x1 + (14 if x2 > x1 else -14), y, n, CONTROL if dashed else "#3D4756")
        f.text(left + 30 if x2 > x1 else left + 12, y - 7, text, 11.5, fill="#1F2933", layer="label")
        if note:
            f.text(left + 30 if x2 > x1 else left + 12, y + 15, note, 10.5, fill=MUTED, layer="label")

    msg(1, "room", "api", 136, "tube light off: darker frames", note="GET /shot.jpg every 5 s")
    msg(2, "api", "api", 178, "unseen-lighting share > 50%", note="window of 120 frames vs training histogram")
    msg(3, "prom", "api", 222, "scrape /metrics (15 s)", note="roomsense_drift_ood_share")
    msg(4, "prom", "am", 264, "RoomSenseDrift, after 5 min", dashed=True)
    msg(5, "am", "you", 300, "alert message: firing", dashed=True)
    msg(6, "am", "api", 352, "POST /hooks/alertmanager", dashed=True, note="retrain flag raised, saved to disk")
    msg(7, "legion", "api", 396, "GET /retrain  →  requested: true", dashed=True, note="retrain.py --watch polls")
    msg(8, "legion", "legion", 438, "git pull, dvc pull, dvc repro", note="only changed stages rerun")
    msg(9, "legion", "mlflow", 482, "log run, register vN+1", note="promote only if better")
    msg(10, "legion", "api", 534, "POST /retrain/ack  (flag cleared)", dashed=True)
    msg(11, "api", "mlflow", 574, "alias moved? download ONNX, swap the model in place", note="poll every 60 s, and right after the ack")
    msg(12, "api", "api", 616, "new reference covers the dark room: share drops")
    msg(13, "prom", "am", 650, "alert resolved", dashed=True)
    msg(14, "am", "you", 682, "alert message: resolved", dashed=True)
    f.legend(28, 34, roles=("serve", "monitor", "train", "store", "ext"), lines=True, steps=True)
    return f



# 5 --------------------------------------------------------------------------
def deployment():
    f = Fig(1200, 690, "RoomSense deployment view")
    f.lane(16, 44, 196, 630, "Tailnet", "lu-globe-lock", "private")
    f.lane(226, 44, 958, 630, "nova", "lu-server", "Ubuntu 26.04 · every published port bound to 127.0.0.1")
    f.add("lane", '<rect x="560" y="84" width="610" height="524" rx="12" fill="#EEF5FD" stroke="#6C9BD9" stroke-width="1.2"/>')
    f.icon(572, 92, 18, "docker", layer="lane")
    f.text(596, 106, "Docker network  roomsense", 12.5, bold=True, fill="#2F6DB5", layer="lane")
    f.text(770, 106, "deploy/compose.yaml  +  capture/compose.yaml", 11, fill=MUTED, layer="lane")

    f.box("dev", 30, 96, 168, 54, "Your devices", ["phone · laptop"], "ext", icon="lu-smartphone")
    f.box("legion", 30, 424, 168, 66, "Legion", ["to MLflow :5443", "and API :4443/api"], "train", icon="lu-cpu")
    f.box("ts", 244, 84, 150, 524, "", [], "ext")
    f.icon(256, 94, 20, "tailscale")
    f.text(282, 108, "Tailscale Serve", 12.5, bold=True)
    f.text(256, 128, "HTTPS, WireGuard", 10.5, fill=MUTED)
    f.text(408, 106, "host port", 11, fill=MUTED)
    f.text(408, 120, "127.0.0.1", 11, fill=MUTED, mono=True)

    rows = [("frontend", "frontend", "nginx · web UI", "serve", "nginx", ":4443", ":8200", ":80"),
            ("api", "api", "FastAPI · ONNX Runtime", "serve", "fastapi", "via :4443/api", ":8300", ":8000"),
            ("mlflow", "mlflow", "tracking + registry · sqlite", "store", "mlflow", ":5443", ":5000", ":5000"),
            ("grafana", "grafana", "dashboards", "monitor", "grafana", ":3443", ":3300", ":3000"),
            ("prometheus", "prometheus", "metrics · alert rules", "monitor", "prometheus", ":6443", ":9091", ":9090"),
            ("alertmanager", "alertmanager", "Telegram · webhook", "monitor", None, "", ":9093", ":9093"),
            ("capture", "roomsense-capture", "FastAPI · own compose file", "data", "fastapi", ":7443", ":8100", ":8100")]
    y = 140
    for bid, name, sub, role, logo, serve, host, cont in rows:
        f.box(bid, 576, y, 300, 48, name, [sub], role, logo=logo, icon=None if logo else "lu-bell", title_size=12.5)
        f.text(866, y + 29, cont, 11.5, mono=True, anchor="end", fill="#374151")
        f.chip(416, y + 14, "127.0.0.1" + host, "ext", size=10)
        if serve.startswith(":"):
            f.chip(262, y + 14, "https " + serve, "ext", size=10)
            f.arrow([(262 + len("https " + serve) * 6.15 + 18, y + 24), (414, y + 24)], width=1.2)
        elif serve:
            f.text(262, y + 28, serve, 10.5, fill=MUTED)
        f.arrow([(416 + len("127.0.0.1" + host) * 6.15 + 18, y + 24), (574, y + 24)], width=1.2)
        y += 62
    # exporters: internal only
    f.box("ne", 900, 140, 256, 48, "node-exporter", ["host metrics · :9100"], "monitor", icon="lu-gauge", title_size=12.5)
    f.box("ca", 900, 202, 256, 48, "cadvisor", ["container metrics · :8080"], "monitor", icon="lu-gauge", title_size=12.5)
    f.box("nx", 900, 264, 256, 48, "nginx-exporter", ["frontend requests · :9113"], "monitor", icon="lu-gauge",
          title_size=12.5)
    f.box("ms", 900, 326, 256, 48, "mlflow-serve", ["optional profile · :5001"], "store", logo="mlflow", dashed=True,
          title_size=12.5)
    f.text(900, 132, "no published port (internal only)", 10.5, fill=MUTED)
    f.box("vol", 900, 400, 256, 194, "", [], "store")
    f.icon(912, 410, 18, "lu-hard-drive", "#8C7313")
    f.text(936, 424, "Volumes (bind mounts)", 12.5, bold=True)
    for i, line in enumerate(["deploy/data/mlflow   runs, models", "deploy/data/api      retrain flag, events",
                              "deploy/data/prometheus   60 days", "deploy/data/grafana", "deploy/data/alertmanager",
                              "data/raw             captured frames", "/mnt/photos/dvc-store   Seagate"]):
        f.text(912, 448 + i * 20, line, 9.8, mono=True, fill="#374151")

    f.arrow([f.anchor("dev", "r"), (228, 123), (228, 164), (244, 164)], label="HTTPS", lpos=0.3, ldy=-10, label_bg=False)
    f.arrow([f.anchor("legion", "r"), (228, 457), (228, 288), (244, 288)] )
    f.text(244, 636, "Not reachable from the room Wi-Fi or the internet: ufw denies incoming traffic, Tailscale Funnel is off,",
           11.5, fill="#374151")
    f.text(244, 654, "and Docker publishes only on 127.0.0.1. Inside the tailnet, traffic is WireGuard-encrypted end to end.",
           11.5, fill="#374151")
    f.legend(28, 34, roles=("data", "serve", "monitor", "store", "train", "ext"), lines=False)
    return f


# 6a -------------------------------------------------------------------------
def lld_request_flow():
    f = Fig(1200, 520, "LLD: request and response flow of POST /predict and the camera loop")
    f.lane(16, 44, 1168, 250, "POST /api/predict", "lu-image", "an uploaded photo, synchronous")
    f.lane(16, 308, 1168, 196, "Camera loop", "lu-camera", "background thread, every 5 s")
    y = 104
    f.box("br", 30, y, 124, 62, "Browser", ["app.js", "FormData(file)"], "ext", icon="lu-monitor")
    f.box("ng", 184, y, 130, 62, "nginx", ["/api/ → API", "max 15 MB"], "serve", logo="nginx")
    f.box("mw", 344, y, 140, 62, "Middleware", ["count + time", "the request"], "serve", logo="fastapi")
    f.box("val", 514, y, 140, 62, "predict()", ["validate file,", "size, truth label"], "serve")
    f.box("dec", 684, y, 140, 62, "frames", ["decode_jpeg()", "to_input() 224²"], "data", icon="lu-image")
    f.box("mod", 854, y, 150, 62, "Model.predict()", ["ONNX Runtime", "softmax"], "serve", logo="onnx")
    f.box("js", 1034, y, 136, 62, "JSON 200", ["label, confidence,", "probabilities"], "store", icon="lu-file-text")
    for n, (a, b) in enumerate([("br", "ng"), ("ng", "mw"), ("mw", "val"), ("val", "dec"), ("dec", "mod"), ("mod", "js")], 1):
        f.link(a, b, step=n, step_at=0.5)
    f.arrow([f.anchor("js", "b"), (1102, 216), (92, 216), f.anchor("br", "b")], label="response travels back the same way",
            lpos=0.5, width=1.3)
    errs = [(514, "400 empty or not an image"), (514, "413 larger than 15 MB"), (684, "422 unknown truth label"),
            (854, "503 no model loaded"), (184, "503 API not answering")]
    ex = 184
    f.text(30, 262, "Errors (JSON {detail}):", 11, bold=True, fill="#9B2C2C")
    ex = 176
    for _x, text in sorted(errs, key=lambda e: e[1]):
        f.add("node", f'<rect x="{ex}" y="248" width="{len(text) * 6.2 + 18:.0f}" height="20" rx="10" fill="#FDECEC" stroke="#D98080"/>')
        f.text(ex + 9, 262, text, 10.5, fill="#9B2C2C")
        ex += len(text) * 6.2 + 28
    f.text(30, 284, "?monitor_frame=true sends the frame through Monitor.observe() instead (the drift demo), exactly like a camera frame.",
           11, fill=MUTED)

    y = 368
    f.box("cam", 30, y, 140, 62, "Camera", ["GET /shot.jpg", "timeout 5 s"], "ext", icon="lu-smartphone")
    f.box("once", 204, y, 150, 62, "camera_once()", ["decode, set", "last-frame time"], "serve")
    f.box("truth", 204, y + 76, 150, 44, "capture /status", ["truth label, if recording"], "data", title_size=11.5)
    f.box("obs", 392, y, 160, 62, "Monitor.observe()", ["predict + brightness"], "serve")
    f.box("win", 590, y, 150, 62, "drift.Window", ["120 × (brightness,", "confidence, correct)"], "serve")
    f.box("met", 778, y, 180, 62, "Prometheus metrics", ["latency, confidence,", "OOD share, accuracy"], "monitor",
          logo="prometheus")
    f.box("lat", 990, y, 180, 62, "latest + latest.jpg", ["served by GET /latest", "read by the Live page"], "store",
          icon="lu-file-text")
    for n, (a, b) in enumerate([("cam", "once"), ("once", "obs"), ("obs", "win"), ("win", "met")], 1):
        f.link(a, b, step=n, step_at=0.5)
    f.arrow([f.anchor("truth", "r"), (472, 466), f.anchor("obs", "b")], width=1.3)
    f.arrow([f.anchor("obs", "t", 40), (512, 350), (1078, 350), f.anchor("lat", "t")], step=5, step_at=0.5, width=1.3)
    f.text(590, 470, "camera error → counted, logged once, the loop keeps running", 11, fill="#9B2C2C")
    f.legend(28, 34, roles=("data", "serve", "monitor", "store", "ext"), lines=False, steps=True)
    return f


# 6b -------------------------------------------------------------------------
def lld_modules():
    f = Fig(1200, 716, "LLD: main modules and classes")

    def cls(bid, x, y, w, title, lines, role, sub=""):
        h = 30 + len(lines) * 15 + 8
        fill, stroke, _ = __import__("figlib").ROLES[role]
        f.boxes[bid] = (x, y, w, h)
        f.add("node", f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="{stroke}" stroke-width="1.4"/>'
                      f'<path d="M{x},{y + 26} h{w}" stroke="{stroke}" stroke-width="1"/>'
                      f'<path d="M{x + 8},{y} h{w - 16} a8,8 0 0 1 8,8 v18 h-{w} v-18 a8,8 0 0 1 8,-8z" fill="{fill}"/>')
        f.text(x + 10, y + 18, title, 12, bold=True)
        if sub:
            f.text(x + w - 8, y + 18, sub, 9.5, fill=MUTED, anchor="end")
        for i, line in enumerate(lines):
            f.text(x + 10, y + 43 + i * 15, line, 9.8, mono=True, fill="#374151")
        return h

    f.lane(16, 44, 560, 344, "deploy/api/app", "lu-package", "the inference API (FastAPI)")
    f.lane(590, 44, 594, 344, "roomsense/", "lu-package", "shared library: training and serving use the same code")
    f.lane(16, 402, 560, 298, "scripts/", "lu-workflow", "DVC stages and operations (Legion)")
    f.lane(590, 402, 290, 298, "capture/", "lu-camera", "data ingestion (nova)")
    f.lane(894, 402, 290, 298, "deploy/frontend/site", "lu-monitor", "static pages (nginx)")

    cls("main", 30, 84, 170, "main.py", ["app = FastAPI()", "17 route functions", "record_request()", "unhandled_error()"], "serve", "module")
    cls("mm", 216, 84, 170, "ModelManager", ["load(force) -> bool", "current() -> Model", "poll_forever()", "_registry_target()"], "serve", "model.py")
    cls("model", 402, 84, 160, "Model", ["predict(img) -> dict", "info() -> dict", "session, classes,", "reference"], "serve", "model.py")
    cls("mon", 30, 196, 170, "Monitor", ["observe(img, source)", "drift_summary()", "camera_once()", "camera_forever()"], "serve", "monitor.py")
    cls("ps", 216, 196, 170, "PipelineStore", ["refresh() -> dict", "snapshot(max_age)", "poll_forever()", "_export_metrics()"], "serve", "pipeline.py")
    cls("st", 402, 196, 160, "RetrainState", ["request(reason)", "ack(detail)", "snapshot()"], "serve", "state.py")
    cls("ev", 402, 290, 160, "EventLog", ["add(kind, msg, level)", "recent(limit)"], "serve", "state.py")
    cls("mt", 30, 308, 170, "metrics.py", ["36 Prometheus metrics", "(Counter/Gauge/Histogram)"], "monitor", "module")
    cls("se", 216, 308, 170, "settings.py", ["every setting from", "environment variables"], "ext", "module")

    cls("fr", 604, 84, 180, "frames", ["decode_jpeg(bytes)", "to_input(img, size)", "brightness(img)", "softmax(logits)"], "data", "module")
    cls("dr", 798, 84, 180, "drift", ["Window.add / snapshot", "ood_share(values, ref)", "psi(ref, cur)", "reference_stats()", "summarise(window, ref)"], "data", "module + class")
    cls("rg", 992, 84, 178, "registry", ["list_versions(client)", "promote(client, v)", "rollback(client, to)", "rollback_target()"], "store", "module")
    cls("rl", 604, 196, 180, "runlog", ["start_stage(name)", "StageRun.done(items)", "StageRun.fail(error)", "provenance()"], "train", "module + class")
    cls("em", 798, 212, 180, "evalmetrics", ["macro_f1_present()", "per_class_f1()"], "train", "module")
    cls("ca", 992, 196, 178, "calibration", ["fit_temperature()", "ece(probs, labels)"], "train", "module")
    cls("mp", 604, 308, 180, "model_pkg", ["load_package(path)", "-> (onnx_path, meta)"], "store", "module")
    cls("oe", 798, 296, 180, "onnx_eval", ["session(path)", "macro_f1(sess, df)"], "train", "module")

    cls("s1", 30, 442, 170, "prepare · split", ["check_row()  prepare()", "assign_sessions()"], "train", "stages 1, 3")
    cls("s2", 216, 442, 170, "preprocess", ["@ray.remote process_chunk", "@ray.remote pick"], "train", "stage 2")
    cls("s3", 402, 442, 160, "train · export", ["sample_weights()", "Calibrated(nn.Module)"], "train", "stages 4, 5")
    cls("s4", 30, 536, 170, "register", ["promotion gate", "registry.promote()"], "train", "stage 6")
    cls("s5", 216, 536, 170, "run_pipeline", ["parse_dag()  run_dvc()", "publish() -> MLflow run"], "train", "runner")
    cls("s6", 402, 536, 160, "retrain · rollback", ["retrain(reason)", "registry.rollback()"], "train", "ops")
    cls("s7", 30, 616, 356, "test_report", ["run_pytest()  offline_criteria()  live_criteria()"], "ext", "tests + acceptance")

    cls("c1", 604, 442, 262, "app.py", ["capture_loop()", "/session/start  /session/stop", "/status  /stats  /metrics"], "data", "FastAPI")
    cls("c2", 604, 536, 262, "storage.py", ["append_frame()  validate_combo()", "trash_frames()  restore()  relabel()", "dataset_stats()"], "data", "lock-guarded CSV")
    cls("c3", 604, 632, 262, "review.py · capmetrics.py", ["review tool routes, metrics"], "data")

    cls("f1", 908, 442, 262, "common.js", ["api(path) -> JSON or Error", "refreshStatus()  setupHeader()"], "serve", "REST client")
    cls("f2", 908, 520, 262, "app.js", ["refreshLive()  refreshDrift()", "upload: validate, POST /predict"], "serve", "Live page")
    cls("f3", 908, 598, 262, "pipeline.js · config.js", ["renderOverview()  act(path)", "ROOMSENSE_CONFIG.apiBase"], "serve", "Pipeline page")

    f.legend(28, 34, roles=("data", "train", "serve", "monitor", "store", "ext"), lines=False)
    f.text(600, 34, "Every script and the API import roomsense/: frames, drift and registry are shared, not copied.",
           11, fill=MUTED)
    return f



FIGURES = {"01-architecture": architecture, "02-pipeline": pipeline, "03-serving-monitoring": serving,
           "04-drift-retrain-sequence": drift_sequence, "05-deployment": deployment,
           "06a-lld-request-flow": lld_request_flow, "06b-lld-modules": lld_modules}


def main():
    wanted = sys.argv[1:] or list(FIGURES)
    for name in wanted:
        FIGURES[name]().save(name)
        print("wrote", name)


if __name__ == "__main__":
    main()
