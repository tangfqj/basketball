"""Metrics for the bib reader (Milestone 2 step 3; requirements §11.3).

Works on per-crop class probabilities (rows of index.csv from training/bib_dataset.py + a probability
matrix). Two levels:
  * crop:   top-1 on un-occluded crops, and accuracy / coverage when only confident crops are kept;
  * window: a vote over all crops of one player in a time window (like a track fragment in the pipeline),
            without and with the roster restriction (only the numbers of the player's team in that video).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

CLASS_PREFIX = "b"   # not "n0..": Ultralytics would read that as an ImageNet class code


def class_name(number: int) -> str:
    return f"{CLASS_PREFIX}{int(number):02d}"


def class_number(name: str) -> int:
    return int(name[len(CLASS_PREFIX):])


def vote(probs: np.ndarray, allowed: list[int] | None = None, min_conf: float = 0.5) -> tuple[int | None, float]:
    """Confidence-weighted vote over the crops of one player -> (class index or None, share of the winner).
    `allowed`: class indices to choose from (roster); probabilities are renormalised over them.
    Crops whose best (allowed) probability is below `min_conf` do not vote."""
    if len(probs) == 0:
        return None, 0.0
    p = probs if allowed is None else probs[:, allowed]
    p = p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
    p = p[p.max(1) >= min_conf]
    if not len(p):
        return None, 0.0
    s = p.sum(0)
    k = int(np.argmax(s))
    return (k if allowed is None else allowed[k]), float(s[k] / s.sum())


def evaluate(rows: list[dict], probs: np.ndarray, classes: list[str], max_occl: float = 0.3,
             window_frames: int = 90, min_conf: float = 0.5) -> dict:
    """rows: index.csv rows (video, frame, track, team, number, occl); probs: (N, C) in `classes` order."""
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx.get(class_name(r["number"]), -1) for r in rows])
    occl = np.array([float(r["occl"]) for r in rows])
    pred, conf = probs.argmax(1), probs.max(1)
    clean = occl <= max_occl
    out = {"crops": len(rows), "crops_clean": int(clean.sum()),
           "crop_acc_clean": round(float((pred == y)[clean].mean()), 4) if clean.any() else None,
           "crop_acc_all": round(float((pred == y).mean()), 4) if len(y) else None}
    for t in (0.5, 0.8, 0.95):
        m = clean & (conf >= t)
        out[f"crop_acc_conf{t}"] = round(float((pred == y)[m].mean()), 4) if m.any() else None
        out[f"crop_coverage_conf{t}"] = round(float(m.mean()), 4) if len(m) else None

    roster = defaultdict(set)      # (video, team) -> class indices
    for r, k in zip(rows, y):
        if k >= 0:
            roster[(r["video"], r["team"])].add(int(k))
    groups = defaultdict(list)     # (video, track, window) -> row indices
    for i, r in enumerate(rows):
        groups[(r["video"], r["track"], int(r["frame"]) // window_frames)].append(i)
    res = {"free": [0, 0, 0], "roster": [0, 0, 0]}   # correct, wrong, undecided
    per_number = defaultdict(lambda: [0, 0])
    for (v, _, _), ii in groups.items():
        ii = np.array(ii)
        truth = y[ii[0]]
        for mode, allowed in (("free", None), ("roster", sorted(roster[(v, rows[ii[0]]["team"])]))):
            k, _ = vote(probs[ii], allowed, min_conf)
            res[mode][2 if k is None else (0 if k == truth else 1)] += 1
            if mode == "roster":
                per_number[classes[truth] if truth >= 0 else "?"][0 if k == truth else 1] += 1
    for mode, (c, w, u) in res.items():
        n = c + w + u
        out[f"window_acc_{mode}"] = round(c / n, 4) if n else None
        out[f"window_undecided_{mode}"] = round(u / n, 4) if n else None
    out["windows"] = len(groups)
    out["window_s"] = round(window_frames / 29.97, 1)
    out["roster_acc_per_number"] = {k: round(a / (a + b), 3) for k, (a, b) in sorted(per_number.items())}
    return out
