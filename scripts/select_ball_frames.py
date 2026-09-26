"""Pick frames for ball annotation and extract them at full resolution (Phase 2.2).

Per video (~120 frames):
  * `flight`: frames in windows where the ball flies towards the rim — from shot labels when available
    (labels/<video>.csv), otherwise from peaks of data/ball_frames/<video>/flight.csv;
  * `random`: frames spread over the game (dribbling, passing, ball in hands, no ball visible).
Validation split: the last 20% of the game time (neighbouring frames never straddle train/val).

  uv run python scripts/select_ball_frames.py IMG_0105
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from hoopstats.datasets.trackid3x3 import load_delimitation
from hoopstats.video import probe

GT = Path("data/trackid3x3_repo/ground_truth/Outdoor/delimitation_frames")


def flight_windows(video: str, fps: float) -> list[tuple[float, float]]:
    labels = Path("labels") / f"{video}.csv"
    if labels.exists():
        with open(labels, newline="") as fh:
            return [(float(r["timestamp_s"]) - 0.1, float(r["timestamp_s"]) + 1.5) for r in csv.DictReader(fh)]
    d = np.loadtxt(Path("data/ball_frames") / video / "flight.csv", delimiter=",", skiprows=1)
    t, k = d[:, 0], np.convolve(d[:, 1], np.ones(5), "same")
    peaks: list[float] = []
    for i in np.argsort(-k):
        if k[i] < 30:
            break
        if all(abs(t[i] - p) > 3 for p in peaks):
            peaks.append(float(t[i]))
    return [(p - 1.3, p + 0.4) for p in sorted(peaks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n-flight", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    src = Path("data/outdoor") / f"{a.video}.MOV"
    out = Path("data/ball_frames") / a.video
    info = probe(src)
    fps = info.fps
    flow = load_delimitation(GT / f"{a.video}.csv")
    start = next((e.frame for e in flow if e.label == "Game start"), 0) / fps
    end = next((e.frame for e in flow if e.label == "Game end"), info.n_frames) / fps
    val_from = start + (1 - a.val_frac) * (end - start)

    wins = [(max(w0, start), min(w1, end)) for w0, w1 in flight_windows(a.video, fps) if w1 > start and w0 < end]
    per = max(1, math.ceil(a.n_flight / max(1, len(wins))))
    picks = []
    for w0, w1 in wins:
        for t in np.sort(rng.uniform(w0, w1, per)):
            picks.append((t, "flight"))
    rng.shuffle(picks)
    picks = picks[: a.n_flight]
    picks += [(t, "random") for t in rng.uniform(start, end, a.n_random)]

    chosen: dict[int, dict] = {}
    for t, kind in sorted(picks):
        f = round(t * fps)
        if any(abs(f - g) < 5 for g in chosen):       # avoid near-duplicates
            continue
        chosen[f] = {"frame": f, "t": round(f / fps, 3), "kind": kind, "split": "val" if t >= val_from else "train"}

    cap = cv2.VideoCapture(str(src))
    (out / "images").mkdir(parents=True, exist_ok=True)
    for f in sorted(chosen):
        p = out / "images" / f"{a.video}_f{f:05d}.jpg"
        if p.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    items = sorted(chosen.values(), key=lambda r: r["frame"])
    (out / "frames.json").write_text(json.dumps(items, indent=1))
    n = {k: sum(r["kind"] == k for r in items) for k in ("flight", "random")}
    print(a.video, f"windows {len(wins)}", n, "val", sum(r["split"] == "val" for r in items), "of", len(items))


if __name__ == "__main__":
    main()
