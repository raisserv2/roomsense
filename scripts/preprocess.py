import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

P = yaml.safe_load(open("params.yaml"))["preprocess"]
RAW, FINAL, TMP = "data/raw", "data/processed/images", "data/_interim"
SIZE = P["img_size"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


@ray.remote
def process_chunk(items, raw_root, out_root, size, blur):
    cv2.setNumThreads(1)
    front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prof = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    res = []
    for rel in items:
        img = cv2.imread(os.path.join(raw_root, rel))
        if img is None:
            res.append((rel, -1))
            continue
        nf = 0
        if blur:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            W = gray.shape[1]
            boxes = [tuple(b) for b in front.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))]
            boxes += [tuple(b) for b in prof.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))]
            for (x, y, w, h) in prof.detectMultiScale(cv2.flip(gray, 1), 1.1, 4, minSize=(24, 24)):
                boxes.append((W - x - w, y, w, h))
            for (x, y, w, h) in boxes:
                p = int(0.3 * max(w, h))
                x0, y0 = max(0, x - p), max(0, y - p)
                x1, y1 = min(img.shape[1], x + w + p), min(img.shape[0], y + h + p)
                k = max(3, (max(x1 - x0, y1 - y0) // 2) | 1)
                img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (k, k), 0)
            nf = len(boxes)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        res.append((rel, nf))
    return res


@ray.remote
def pick(emb, k, seed):
    n = len(emb)
    if k >= n:
        return list(range(n))
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
    keep = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
        keep.append(int(members[d.argmin()]))
    return sorted(keep)


tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class DS(Dataset):
    def __init__(self, p):
        self.p = p

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return tf(Image.open(os.path.join(TMP, self.p[i])).convert("RGB"))


df = pd.read_csv("data/processed/manifest.csv")
df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
n_in = len(df)

# 1. trim the first/last frames of every session
t = P["trim_n"]
df = pd.concat([g.iloc[t:len(g) - t] if len(g) > 2 * t + 2 else g
                for _, g in df.groupby("session_id", sort=True)]).reset_index(drop=True)
n_trim = len(df)

# 2. blur faces + resize, in parallel
ray.init(include_dashboard=False, logging_level="ERROR")
paths = df.path.tolist()
futs = [process_chunk.remote(paths[i:i + 200], RAW, TMP, SIZE, P["blur_faces"])
        for i in range(0, len(paths), 200)]
res = dict(r for part in ray.get(futs) for r in part)
df["faces_blurred"] = df.path.map(res)
df = df[df.faces_blurred >= 0].reset_index(drop=True)

# 3. embed every frame (MobileNetV3-Small features)
m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval().to(dev)
embs = []
with torch.no_grad():
    for x in DataLoader(DS(df.path.tolist()), batch_size=128, num_workers=4):
        embs.append(torch.flatten(m.avgpool(m.features(x.to(dev))), 1).float().cpu().numpy())
emb = np.concatenate(embs)

# 4. per-session k-means, keep the frame closest to each cluster centre
groups = df.groupby("session_id").indices
jobs = []
for sid, ix in groups.items():
    n = len(ix)
    k = min(n, int(min(P["max_k"], max(P["min_k"], round(n * P["keep_ratio"])))))
    jobs.append((sid, ix, pick.remote(emb[ix], k, P["seed"])))
sel = {sid: ix[np.array(ray.get(f))] for sid, ix, f in jobs}
keep_idx = np.sort(np.concatenate(list(sel.values())))
out = df.iloc[keep_idx].reset_index(drop=True)

# 5. write the final images + manifest
for rel in out.path:
    dst = os.path.join(FINAL, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(TMP, rel), dst)
cols = ["path", "session_id", "label", "lighting", "timestamp", "period", "faces_blurred"]
out[cols].to_csv("data/processed/clean_manifest.csv", index=False)

# 6. cluster maps for the biggest sessions (figures for the report)
Path("reports/clusters").mkdir(parents=True, exist_ok=True)
biggest = sorted(groups, key=lambda s: -len(groups[s]))[:P["plot_sessions"]]
for sid in biggest:
    ix = groups[sid]
    xy = PCA(n_components=2, random_state=0).fit_transform(emb[ix])
    kept = np.isin(ix, sel[sid])
    plt.figure(figsize=(6, 5))
    plt.scatter(xy[~kept, 0], xy[~kept, 1], s=8, c="lightgrey", label="dropped")
    plt.scatter(xy[kept, 0], xy[kept, 1], s=30, c="crimson", label="kept")
    plt.title(f"{sid}: kept {kept.sum()} of {len(ix)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/clusters/{sid}.png", dpi=110)
    plt.close()

report = {
    "frames_in": n_in, "after_trim": n_trim, "kept": len(out),
    "sessions": len(groups), "faces_blurred_total": int(out.faces_blurred.sum()),
    "kept_per_class": out.label.value_counts().to_dict(),
}
json.dump(report, open("reports/preprocess_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
shutil.rmtree(TMP, ignore_errors=True)
