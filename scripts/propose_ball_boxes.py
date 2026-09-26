"""Pre-annotate ball boxes with the pretrained detector (for the review tool). Resumable.

Two passes per frame (as in the probe): full frame at 1920 px, and a 1280x960 native-resolution crop
around the rim. Candidates from both are merged (IoU > 0.3 -> keep the higher score), top 3 kept.

  uv run python scripts/propose_ball_boxes.py IMG_0104 --budget-s 140
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from hoopstats.calibration import Calibration

BALL = 32


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="data/models/yolo11s.pt")
    ap.add_argument("--budget-s", type=float, default=140)
    a = ap.parse_args()
    d = Path("data/ball_frames") / a.video
    out_p = d / "proposals.json"
    props = json.loads(out_p.read_text()) if out_p.exists() else {}
    cal = Calibration.load(f"calib/{a.video}.json")
    model = YOLO(a.model)
    t0 = time.time()
    frames = json.loads((d / "frames.json").read_text())
    for r in frames:
        name = f"{a.video}_f{r['frame']:05d}.jpg"
        if name in props:
            continue
        if time.time() - t0 > a.budget_s:
            break
        img = cv2.imread(str(d / "images" / name))
        H, W = img.shape[:2]
        rx0 = int(np.clip(cal.rim_center[0] - 640, 0, W - 1280))
        ry0 = int(np.clip(cal.rim_center[1] - 600, 0, H - 960))
        cands = []
        small = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        for src, im, s, dx, dy, sz in (("full", small, W / 1920, 0, 0, 1920),
                                       ("rim", img[ry0:ry0 + 960, rx0:rx0 + 1280], 1.0, rx0, ry0, 1280)):
            res = model.predict(im, imgsz=sz, conf=0.05, classes=[BALL], verbose=False)[0]
            for (x0, y0, x1, y1), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                cands.append({"conf": round(c, 3), "box": [round(x0 * s + dx, 1), round(y0 * s + dy, 1),
                              round((x1 - x0) * s, 1), round((y1 - y0) * s, 1)], "src": src})
        kept = []
        for c in sorted(cands, key=lambda c: -c["conf"]):
            if all(iou(c["box"], k["box"]) <= 0.3 for k in kept):
                kept.append(c)
        props[name] = kept[:3]
        out_p.write_text(json.dumps(props))
    n_with = sum(bool(v) for v in props.values())
    print(f"{a.video}: {len(props)}/{len(frames)} frames proposed, {n_with} with a candidate, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
