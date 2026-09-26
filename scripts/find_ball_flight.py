"""Scan a video for moments when an orange object moves near/above the rim (ball in flight).

Cheap signal used to pick shot-like frames in videos without shot labels (ball-annotation sampling).
Writes data/ball_frames/<video>/flight.csv (time_s, score); resumable in chunks (--budget-s).

  uv run python scripts/find_ball_flight.py IMG_0105
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.video import probe

FPS_OUT = 10          # analysed frames per second
SCALE = 0.5
HALF_W, UP, DOWN = 450, 500, 150   # ROI around the rim centre, 4K pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--chunk-s", type=float, default=60)
    ap.add_argument("--budget-s", type=float, default=140)
    a = ap.parse_args()
    src = Path("data/outdoor") / f"{a.video}.MOV"
    out = Path("data/ball_frames") / a.video
    out.mkdir(parents=True, exist_ok=True)
    cal = Calibration.load(f"calib/{a.video}.json")
    cx, cy = (int(v) for v in cal.rim_center)
    x0, y0, w, h = cx - HALF_W, cy - UP, 2 * HALF_W, UP + DOWN
    ow, oh = int(w * SCALE), int(h * SCALE)
    dur = probe(src).duration_s
    t_start = time.time()
    for c0 in np.arange(0, dur, a.chunk_s):
        part = out / f"flight_{int(c0):04d}.csv"
        if part.exists():
            continue
        if time.time() - t_start > a.budget_s:
            print("budget used; re-run to continue")
            return
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{c0}", "-t", f"{a.chunk_s}", "-i", str(src),
               "-vf", f"fps={FPS_OUT},crop={w}:{h}:{x0}:{y0},scale={ow}:{oh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, oh, ow, 3)
        rows, prev = [], None
        for i, f in enumerate(frames):
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            orange = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 20) & (hsv[..., 1] >= 130) & (hsv[..., 2] >= 80)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16)
            moving = np.abs(g - prev) > 25 if prev is not None else np.zeros_like(orange)
            prev = g
            rows.append(f"{c0 + i / FPS_OUT:.2f},{int((orange & moving).sum())}\n")
        part.write_text("".join(rows))
        print(f"{a.video}: {c0:.0f}-{c0 + a.chunk_s:.0f} s done", flush=True)
    parts = sorted(out.glob("flight_*.csv"))
    (out / "flight.csv").write_text("time_s,score\n" + "".join(p.read_text() for p in parts))
    print("complete:", out / "flight.csv")


if __name__ == "__main__":
    main()
