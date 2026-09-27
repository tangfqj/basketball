"""Train and evaluate the bib reader (Milestone 2 step 3). Used by notebooks/train_bib_reader.ipynb.

One run = one fold: train on the crops of every video except `--held`, then predict all crops of the
held-out video and write per-crop probabilities + metrics. `--held none` trains the final model on all
videos (no evaluation).

  uv run python scripts/train_bib_reader.py --crops data/bib_crops --videos IMG_0104 ... --held IMG_0108
Outputs in --out/<run>/: weights (best.pt), preds_<held>.csv, metrics_<held>.json.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from hoopstats.training.bib_eval import class_name, evaluate


def read_index(crops: Path, videos: list[str]) -> list[dict]:
    rows = []
    for v in videos:
        with open(crops / v / "index.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                r["number"], r["frame"], r["track"] = int(r["number"]), int(r["frame"]), int(r["track"])
                r["path"] = str((crops / v / r["file"]).resolve())
                rows.append(r)
    # the dataset has a few frames with two boxes under one track id (IMG_0105): both rows point to the same
    # file, and it is unknown which box it shows -> drop them
    n = collections.Counter(r["path"] for r in rows)
    dup = sum(k > 1 for k in n.values())
    if dup:
        print(f"dropping {dup} ambiguous crops (two boxes with one track id in the ground truth)")
    return [r for r in rows if n[r["path"]] == 1]


def build_split(rows, held, classes, root: Path, max_occl: float, val_frac: float) -> dict:
    shutil.rmtree(root, ignore_errors=True)
    for part in ("train", "val"):
        for c in classes:
            (root / part / c).mkdir(parents=True, exist_ok=True)
    last = {}
    for r in rows:
        last[r["video"]] = max(last.get(r["video"], 0), r["frame"])
    n = {"train": 0, "val": 0}
    for r in rows:
        if r["video"] == held or float(r["occl"]) > max_occl:
            continue
        part = "val" if r["frame"] >= (1 - val_frac) * last[r["video"]] else "train"
        os.symlink(r["path"], root / part / class_name(r["number"]) / r["file"])
        n[part] += 1
    return n


def predict(model, rows, classes, imgsz: int, batch: int = 256) -> np.ndarray:
    col = {name: classes.index(name) for name in model.names.values()}
    order = [col[model.names[i]] for i in range(len(model.names))]
    out = np.zeros((len(rows), len(classes)), np.float32)
    for s in range(0, len(rows), batch):
        res = model.predict([r["path"] for r in rows[s:s + batch]], imgsz=imgsz, verbose=False)
        for k, r in enumerate(res):
            out[s + k, order] = r.probs.data.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/bib_crops")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--held", required=True, help="held-out video, or 'none' for the final model")
    ap.add_argument("--out", default="runs/bib")
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-occl", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1, help="last part of each training video used for validation")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from ultralytics import YOLO

    crops = Path(a.crops)
    rows = read_index(crops, a.videos)
    classes = sorted({class_name(r["number"]) for r in rows})
    run = "final" if a.held == "none" else f"held_{a.held}"
    out = Path(a.out).resolve()
    split = build_split(rows, a.held, classes, out / "datasets" / run, a.max_occl, a.val_frac)
    print(f"{run}: classes {classes}; train {split['train']} / val {split['val']} crops", flush=True)

    model = YOLO(a.model)
    model.train(data=str(out / "datasets" / run), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
                project=str(out), name=run, exist_ok=True, fliplr=0.0, flipud=0.0, scale=0.3, erasing=0.1,
                workers=a.workers, cache="ram", plots=False, verbose=False)
    best = out / run / "weights" / "best.pt"
    shutil.copy(best, out / f"bib_{run}.pt")
    if a.held == "none":
        print(f"final model: {out / f'bib_{run}.pt'}")
        return

    test = [r for r in rows if r["video"] == a.held]
    probs = predict(YOLO(str(best)), test, classes, a.imgsz)
    with open(out / f"preds_{a.held}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "video", "frame", "track", "team", "number", "occl", *classes])
        for r, p in zip(test, probs):
            w.writerow([r["file"], r["video"], r["frame"], r["track"], r["team"], r["number"], r["occl"],
                        *np.round(p, 4)])
    metrics = {"held": a.held, "classes": classes, "train_crops": split["train"], **evaluate(test, probs, classes)}
    (out / f"metrics_{a.held}.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
