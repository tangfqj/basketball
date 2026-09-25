"""Phase 2 feasibility probe: how well does a pretrained COCO YOLO find the ball (and players)?

For every labelled shot, frames from 0.3 s before release to 1.8 s after (every `--step` frames) are
run through the detector in two setups:
  full : whole frame downscaled to 1920 px wide (ball ~ 12-15 px)
  rim  : 1280x960 crop around the rim at native 4K resolution (ball ~ 25-30 px)
Ball detections have no ground truth; we report detection rates and write sample images for visual
checks. Player detections are scored against the TrackID3x3 MOT boxes (recall / precision @ IoU 0.5).

  uv run python scripts/probe_ball_detection.py data/outdoor/IMG_0104.MOV labels/IMG_0104.csv \
      --rim 1825 845 --model yolo11s.pt --out data/_probe/IMG_0104
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from hoopstats.court import get_court, is_on_court
from hoopstats.datasets.trackid3x3 import load_mot, outdoor_calibration

BALL, PERSON = 32, 0


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("labels")
    ap.add_argument("--rim", type=float, nargs=2, required=True, help="rim centre, 4K pixels")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--mot", default="data/trackid3x3_repo/ground_truth/Outdoor/MOT/IMG_0104.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    shots = list(csv.DictReader(open(a.labels)))
    if a.max_shots:
        shots = shots[: a.max_shots]
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    rx0 = int(np.clip(a.rim[0] - 640, 0, W - 1280))
    ry0 = int(np.clip(a.rim[1] - 600, 0, H - 960))

    cal = outdoor_calibration("data/trackid3x3_repo/ground_truth")   # to drop people standing off court
    court = get_court(cal.court_standard)

    def on_court(box):
        (cx, cy), = cal.image_to_court([(box[0] + box[2] / 2, box[1] + box[3])])
        return is_on_court(court, cx, cy, margin=0.3)

    gt = defaultdict(list)
    for d in load_mot(a.mot):
        gt[d.frame].append((d.x, d.y, d.w, d.h))

    rows, t0 = [], time.time()
    for si, s in enumerate(shots):
        f0 = max(0, int(round((float(s["timestamp_s"]) - 0.3) * fps)))
        f1 = int(round((float(s["timestamp_s"]) + 1.8) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for f in range(f0, f1 + 1):
            ok, img = cap.read()
            if not ok:
                break
            if (f - f0) % a.step:
                continue
            small = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
            crop = img[ry0:ry0 + 960, rx0:rx0 + 1280]
            r_full = model.predict(small, imgsz=1920, conf=a.conf, classes=[BALL, PERSON], verbose=False)[0]
            r_rim = model.predict(crop, imgsz=1280, conf=a.conf, classes=[BALL], verbose=False)[0]

            def dets(r, cls, scale=1.0, dx=0, dy=0):
                b = r.boxes
                return [(float(c), [float(x) * scale + dx, float(y) * scale + dy, float(x2 - x) * scale, float(y2 - y) * scale])
                        for (x, y, x2, y2), c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()) if int(k) == cls]

            ball_full = dets(r_full, BALL, 2.0)                  # -> 4K pixels
            ball_rim = dets(r_rim, BALL, 1.0, rx0, ry0)          # -> 4K pixels
            persons = [(c, b) for c, b in dets(r_full, PERSON, 2.0) if on_court(b)]
            g = gt.get(f + 1, [])                                # MOT frames are 1-based
            matched = set()
            tp = 0
            for gb in g:
                best = max(((iou(gb, pb), j) for j, (c, pb) in enumerate(persons) if c >= 0.25 and j not in matched),
                           default=(0, -1))
                if best[0] >= 0.5:
                    tp += 1
                    matched.add(best[1])
            rows.append({"shot": si, "frame": f, "dt": round((f / fps) - float(s["timestamp_s"]), 3),
                         "ball_full": ball_full, "ball_rim": ball_rim, "n_gt_players": len(g), "player_tp": tp,
                         "n_person_dets": sum(c >= 0.25 for c, _ in persons)})
            if (f - f0) % (a.step * 6) == 0:   # a few samples per shot for visual checks
                vis = img.copy()
                cv2.rectangle(vis, (rx0, ry0), (rx0 + 1280, ry0 + 960), (255, 255, 0), 3)
                for c, (x, y, w, h) in ball_full:
                    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 4)
                    cv2.putText(vis, f"F{c:.2f}", (int(x), int(y) - 8), 0, 1.4, (0, 0, 255), 3)
                for c, (x, y, w, h) in ball_rim:
                    cv2.rectangle(vis, (int(x) - 4, int(y) - 4), (int(x + w) + 4, int(y + h) + 4), (0, 255, 0), 4)
                    cv2.putText(vis, f"R{c:.2f}", (int(x), int(y + h) + 40), 0, 1.4, (0, 255, 0), 3)
                cv2.imwrite(str(out / "samples" / f"shot{si:02d}_f{f}.jpg"), cv2.resize(vis, (1920, 1080)))
        print(f"shot {si + 1}/{len(shots)} done, {len(rows)} frames, {time.time() - t0:.0f} s", flush=True)

    (out / "detections.json").write_text(json.dumps(rows))

    def rate(key, thr, cond=lambda r: True):
        sel = [r for r in rows if cond(r)]
        return round(sum(any(c >= thr for c, _ in r[key]) for r in sel) / max(1, len(sel)), 3)

    near_rim = lambda r: r["dt"] >= 0.6   # ball typically travelling to / at the rim  # noqa: E731
    summary = {
        "model": a.model, "frames": len(rows), "shots": len(shots),
        "sec_per_frame": round((time.time() - t0) / max(1, len(rows)), 2),
        "ball_rate_full": {t: rate("ball_full", t) for t in (0.1, 0.25, 0.5)},
        "ball_rate_rim_crop": {t: rate("ball_rim", t) for t in (0.1, 0.25, 0.5)},
        "ball_rate_rim_crop_late": {t: rate("ball_rim", t, near_rim) for t in (0.1, 0.25, 0.5)},
        "shots_with_ball_near_rim": sum(
            any(any(c >= 0.25 for c, _ in r["ball_rim"]) for r in rows if r["shot"] == si and near_rim(r))
            for si in range(len(shots))),
        "player_recall@0.25": round(sum(r["player_tp"] for r in rows) / max(1, sum(r["n_gt_players"] for r in rows)), 3),
        "player_precision@0.25": round(sum(r["player_tp"] for r in rows) / max(1, sum(r["n_person_dets"] for r in rows)), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
