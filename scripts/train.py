import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

P = yaml.safe_load(open("params.yaml"))["train"]
torch.manual_seed(P["seed"])
np.random.seed(P["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/processed/splits.csv")
classes = sorted(df.label.unique())
idx = {c: i for i, c in enumerate(classes)}

tf = T.Compose([
    T.Resize((P["img_h"], P["img_w"])),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DS(Dataset):
    def __init__(self, d):
        self.p = d.path.tolist()
        self.y = [idx[l] for l in d.label]

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open("data/processed/images/" + self.p[i])
        im.draft("RGB", (P["img_w"] * 2, P["img_h"] * 2))  # fast JPEG downscale
        return tf(im.convert("RGB")), self.y[i]


def loader(split, shuffle):
    return DataLoader(DS(df[df.split == split]), batch_size=P["batch_size"],
                      shuffle=shuffle, num_workers=P["workers"],
                      pin_memory=True, persistent_workers=True)


tr, va, te = loader("train", True), loader("val", False), loader("test", False)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.to(dev)

counts = df[df.split == "train"].label.value_counts().reindex(classes).values.astype(float)
w = (counts.sum() / counts) ** 0.5
w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
crit = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=P["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P["epochs"])
scaler = torch.amp.GradScaler()


@torch.no_grad()
def evaluate(dl):
    model.eval()
    ys, ps = [], []
    for x, y in dl:
        with torch.autocast("cuda"):
            out = model(x.to(dev, non_blocking=True))
        ps += out.argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("roomsense")
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

with mlflow.start_run():
    mlflow.log_params(P)
    mlflow.log_param("classes", ",".join(classes))
    best = -1.0
    for ep in range(P["epochs"]):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        ys, ps = evaluate(va)
        f1 = f1_score(ys, ps, average="macro", labels=range(len(classes)), zero_division=0)
        mlflow.log_metrics({"train_loss": tot / len(tr.dataset), "val_macro_f1": f1}, step=ep)
        print(f"epoch {ep + 1}/{P['epochs']}  loss {tot / len(tr.dataset):.4f}  val macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), "models/best.pt")

    model.load_state_dict(torch.load("models/best.pt"))
    ys, ps = evaluate(te)
    labels = list(range(len(classes)))
    metrics = {
        "best_val_macro_f1": best,
        "test_macro_f1": f1_score(ys, ps, average="macro", labels=labels, zero_division=0),
        "test_accuracy": accuracy_score(ys, ps),
        "test_per_class_f1": dict(zip(classes, f1_score(ys, ps, average=None, labels=labels, zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(ys, ps, labels=labels).tolist(),
    }
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2)
    json.dump(classes, open("models/classes.json", "w"))
    mlflow.log_metrics({"test_macro_f1": metrics["test_macro_f1"], "test_accuracy": metrics["test_accuracy"]})
    mlflow.log_artifact("models/best.pt")
    mlflow.log_artifact("reports/metrics.json")
    print(json.dumps(metrics, indent=2))
