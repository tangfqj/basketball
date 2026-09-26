"""Render a debug video of the ball track (and optionally player boxes) for a frame range.

  uv run python scripts/render_ball_track.py IMG_0104 --start 600 --end 900   # -> cache/IMG_0104/track_600_900.mp4
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from hoopstats.detection.runner import load_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--trail", type=int, default=30)
    ap.add_argument("--players", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache_dir) / a.video
    tr = np.load(cache / "ball_track.npz")
    _, persons = load_detections(cache) if a.players else (None, np.zeros((0, 7)))
    cap = cv2.VideoCapture(f"data/outdoor/{a.video}.MOV")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    out_p = cache / f"track_{a.start}_{a.end}.mp4"
    wr = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for f in range(a.start, a.end):
        ok, img = cap.read()
        if not ok:
            break
        for p in persons[persons[:, 0] == f - (f % 3)]:
            x, y, w, h = p[2:6].astype(int)
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 3)
            cv2.putText(img, str(int(p[1])), (x, y - 10), 0, 1.5, (200, 200, 200), 3)
        pts = [(int(tr["x"][g]), int(tr["y"][g]), tr["state"][g]) for g in range(max(0, f - a.trail), f + 1) if tr["state"][g]]
        for (x0, y0, _), (x1, y1, _) in pairwise(pts):
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 255), 4)
        if tr["state"][f]:
            color = (0, 200, 0) if tr["state"][f] == 1 else (0, 0, 255)
            cv2.circle(img, (int(tr["x"][f]), int(tr["y"][f])), 28, color, 5)
        cv2.putText(img, f"frame {f}  t={f / fps:.2f}s", (40, 80), 0, 2.0, (255, 255, 255), 5)
        wr.write(cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA))
    wr.release()
    print(out_p)


if __name__ == "__main__":
    main()
