"""DVC stage `train`: fine-tune MobileNetV3-Small on the session-wise split.

v2.1 "smarter training", built for a small, skewed dataset:
  * class- and session-balanced sampling: every class is seen equally often,
    and one long session cannot dominate its class;
  * label smoothing: the model is never pushed towards 100% certainty;
  * two phases: first only the new classifier head learns (backbone frozen),
    then the whole network fine-tunes at a lower learning rate;
  * colour/brightness augmentation (no flips or crops: the desk cue sits at the
    right edge of the frame);
  * early stopping on validation macro-F1;
  * temperature calibration on the validation set, so "80% sure" means right
    about 80% of the time (baked into the ONNX model by the export stage).

Logs params, per-epoch metrics, test metrics, calibration, a confusion matrix
and the PyTorch model to MLflow ($MLFLOW_TRACKING_URI, else sqlite:///mlflow.db).
CLI overrides (MLproject `train` entry point): --epochs, --lr, --batch-size.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402
from torchvision import models  # noqa: E402
from torchvision import transforms as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense import calibration  # noqa: E402

P = yaml.safe_load(open("params.yaml"))["train"]
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int)
ap.add_argument("--lr", type=float)
ap.add_argument("--batch-size", type=int, dest="batch_size")
for k, v in vars(ap.parse_args()).items():
    if v is not None:
        P[k] = v

torch.manual_seed(P["seed"])
np.random.seed(P["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda"

df = pd.read_csv("data/processed/splits.csv")
classes = sorted(df.label.unique())
idx = {c: i for i, c in enumerate(classes)}

norm = [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
size = (P["img_h"], P["img_w"])
aug = P.get("augment") or {}
train_tf = T.Compose([T.Resize(size),
                      T.ColorJitter(brightness=aug.get("brightness", 0), contrast=aug.get("contrast", 0),
                                    saturation=aug.get("saturation", 0), hue=aug.get("hue", 0)),
                      *norm])
eval_tf = T.Compose([T.Resize(size), *norm])


class DS(Dataset):
    def __init__(self, d, tf):
        self.p = d.path.tolist()
        self.y = [idx[l] for l in d.label]
        self.tf = tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open("data/processed/images/" + self.p[i])
        im.draft("RGB", (P["img_w"] * 2, P["img_h"] * 2))  # fast JPEG downscale
        return self.tf(im.convert("RGB")), self.y[i]


def sample_weights(d: pd.DataFrame) -> np.ndarray:
    """Each class gets the same total weight; inside a class, sessions are evened out
    by (frames in session) ** -session_balance (0 = off, 1 = every session equal)."""
    sess = d.groupby("session_id").path.transform("size").to_numpy(float)
    w = sess ** -float(P.get("session_balance", 0.5))
    lab = d.label.to_numpy()
    for c in np.unique(lab):
        m = lab == c
        w[m] /= w[m].sum()
    return w / w.sum()


tr_df = df[df.split == "train"]
workers = P["workers"]
dl_kw = dict(batch_size=P["batch_size"], num_workers=workers, pin_memory=amp, persistent_workers=workers > 0)
if P.get("balance", "sampler") == "sampler":
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights(tr_df)), num_samples=len(tr_df),
                                    replacement=True, generator=torch.Generator().manual_seed(P["seed"]))
    tr = DataLoader(DS(tr_df, train_tf), sampler=sampler, **dl_kw)
    class_w = None
else:
    tr = DataLoader(DS(tr_df, train_tf), shuffle=True, **dl_kw)
    counts = tr_df.label.value_counts().reindex(classes).fillna(0).values.astype(float)
    cw = (counts.sum() / np.clip(counts, 1, None)) ** 0.5
    class_w = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=dev)
va = DataLoader(DS(df[df.split == "val"], eval_tf), shuffle=False, **dl_kw)
te = DataLoader(DS(df[df.split == "test"], eval_tf), shuffle=False, **dl_kw)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.to(dev)
crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=float(P.get("label_smoothing", 0.0)))
scaler = torch.amp.GradScaler(enabled=amp)


@torch.no_grad()
def predict(dl):
    """Raw logits + labels for a loader."""
    model.eval()
    out, ys = [], []
    for x, y in dl:
        with torch.autocast(dev, enabled=amp):
            out.append(model(x.to(dev, non_blocking=True)).float().cpu())
        ys.append(y)
    if not out:
        return np.zeros((0, len(classes))), np.zeros(0, dtype=int)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(len(classes))), zero_division=0)


def run_epoch(opt, freeze_backbone: bool) -> float:
    model.train()
    if freeze_backbone:
        model.features.eval()  # keep BatchNorm statistics of the frozen backbone
    tot, n = 0.0, 0
    for x, y in tr:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad()
        with torch.autocast(dev, enabled=amp):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("roomsense")
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

with mlflow.start_run(run_name="train") as run:
    mlflow.log_params({k: v for k, v in P.items() if not isinstance(v, dict)})
    mlflow.log_params({f"augment_{k}": v for k, v in aug.items()})
    mlflow.log_param("classes", ",".join(classes))
    mlflow.log_param("train_frames", len(tr_df))
    mlflow.log_dict(yaml.safe_load(open("params.yaml")), "params.yaml")

    best, best_ep, step, stale = -1.0, -1, 0, 0
    phases = [("head", int(P.get("freeze_epochs", 0)), float(P.get("head_lr", P["lr"])), True),
              ("finetune", int(P["epochs"]), float(P["lr"]), False)]
    for phase, n_ep, lr, frozen in phases:
        if n_ep <= 0:
            continue
        for p_ in model.features.parameters():
            p_.requires_grad = not frozen
        params = [p_ for p_ in model.parameters() if p_.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(P.get("weight_decay", 1e-4)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        stale = 0
        for ep in range(n_ep):
            loss = run_epoch(opt, frozen)
            sched.step()
            lv, yv = predict(va)
            f1 = macro_f1(yv, lv.argmax(1)) if len(yv) else 0.0
            mlflow.log_metrics({"train_loss": loss, "val_macro_f1": f1, "lr": sched.get_last_lr()[0]}, step=step)
            print(f"[{phase}] epoch {ep + 1}/{n_ep}  loss {loss:.4f}  val macro-F1 {f1:.4f}", flush=True)
            step += 1
            if f1 > best:
                best, best_ep, stale = f1, step, 0
                torch.save(model.state_dict(), "models/best.pt")
            else:
                stale += 1
                if phase == "finetune" and stale >= int(P.get("patience", 99)):
                    print(f"early stop: no val improvement for {stale} epochs")
                    break

    model.load_state_dict(torch.load("models/best.pt", map_location=dev))
    lv, yv = predict(va)
    lt, yt = predict(te)

    # temperature calibration on validation, checked on test
    temp = calibration.fit_temperature(lv, yv) if P.get("calibrate", True) and len(yv) else 1.0
    pt_raw, pt_cal = calibration.calibrated_probs(lt, 1.0), calibration.calibrated_probs(lt, temp)
    pred = lt.argmax(1)
    labels = list(range(len(classes)))
    cm = confusion_matrix(yt, pred, labels=labels)
    metrics = {
        "best_val_macro_f1": best,
        "best_epoch": best_ep,
        "test_macro_f1": macro_f1(yt, pred),
        "test_accuracy": accuracy_score(yt, pred),
        "test_per_class_f1": dict(zip(classes, f1_score(yt, pred, average=None, labels=labels,
                                                        zero_division=0).tolist())),
        "temperature": temp,
        "test_ece_before_calibration": calibration.ece(pt_raw, yt),
        "test_ece_after_calibration": calibration.ece(pt_cal, yt),
        "test_mean_confidence_before": float(pt_raw.max(1).mean()) if len(yt) else None,
        "test_mean_confidence_after": float(pt_cal.max(1).mean()) if len(yt) else None,
        "confusion_matrix": cm.tolist(),
    }
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2)
    json.dump(classes, open("models/classes.json", "w"))
    json.dump({"temperature": temp}, open("models/calibration.json", "w"), indent=2)
    json.dump({"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
               "tracking_uri": mlflow.get_tracking_uri()}, open("models/train_run.json", "w"), indent=2)

    # confusion matrix figure (rows = truth, columns = prediction)
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.1 * len(classes) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, [c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_yticks(labels, classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title(f"test macro-F1 {metrics['test_macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=120)
    plt.close(fig)

    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None})
    mlflow.log_dict(metrics, "metrics.json")
    mlflow.log_dict({"classes": classes}, "classes.json")
    mlflow.log_artifact("reports/confusion_matrix.png")

    if P.get("log_torch_model", True):
        # Newer MLflow traces PyTorch models ("pt2" format), which needs an example input.
        # A logging hiccup must never throw away a finished training run, so it only warns.
        try:
            example = np.zeros((1, 3, P["img_h"], P["img_w"]), dtype=np.float32)
            mlflow.pytorch.log_model(model.cpu().eval(), name="torch-model", input_example=example)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"PyTorch model not logged to MLflow: {e}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
