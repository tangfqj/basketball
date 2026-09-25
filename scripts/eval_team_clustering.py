"""Evaluate team clustering on TrackID3x3 Outdoor videos (Phase 5).

Frames: keyframes (~1 fps) extracted at 1920 px width into data/_teams/<video>/ (see doc/experiments).
Ground truth: player boxes (MOT) and team membership from the offense/defense lists (delimitation file).

Two modes:
  gt   : cluster the dataset's player boxes (isolates the colour method)
  det  : cluster YOLO person detections filtered to the court (realistic: includes referee etc.);
         detections matched to a GT player (IoU >= 0.5) are scored, unmatched on-court people should be "?"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from hoopstats.court import get_court, is_on_court
from hoopstats.datasets.trackid3x3 import load_delimitation, load_mot, outdoor_calibration, team_membership
from hoopstats.teams import color_feature, estimate_court_hue, fit_team_model, torso_crop, vote

GT_ROOT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
FPS = 30000 / 1001
SCALE = 0.5  # 4K annotations -> 1920 px frames


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih + 1e-9)


def best_mapping(pairs):
    """Accuracy under the better of identity / swapped team names (cluster names are arbitrary)."""
    scored = [(g, p) for g, p in pairs if p != "?"]
    same = sum(g == p for g, p in scored)
    swap = sum(g != p for g, p in scored)
    return max(same, swap) / max(1, len(scored)), swap > same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="e.g. IMG_0104")
    ap.add_argument("--mode", choices=["gt", "det"], default="gt")
    ap.add_argument("--det-frames", type=int, default=60, help="keyframes used in det mode")
    ap.add_argument("--model", default="data/models/yolo11s.pt")
    ap.add_argument("--no-court-mask", action="store_true", help="ablation: keep court-coloured pixels")
    a = ap.parse_args()

    d = Path("data/_teams") / a.video
    index = [(int(n), float(t)) for n, t in (line.split() for line in (d / "index.txt").read_text().splitlines())]
    gt_boxes = defaultdict(list)
    for det in load_mot(GT_ROOT / "MOT" / f"{a.video}.txt"):
        gt_boxes[det.frame].append((det.track_id, (det.x * SCALE, det.y * SCALE, det.w * SCALE, det.h * SCALE)))
    team_of = team_membership(load_delimitation(GT_ROOT / "delimitation_frames" / f"{a.video}.csv"))

    cal = outdoor_calibration(GT_ROOT.parent).scaled(1920, 1080)
    court = get_court(cal.court_standard)

    def on_court(b):
        (x, y), = cal.image_to_court([(b[0] + b[2] / 2, b[1] + b[3])])
        return is_on_court(court, x, y, margin=0.3)

    bg = None if a.no_court_mask else estimate_court_hue(cv2.imread(str(d / "k_0001.jpg")), cal, court)
    print("court hue:", bg)
    samples = []   # (keyframe no, gt team or None, track id or None, feature, sat_frac)
    if a.mode == "gt":
        for n, t in index:
            f = round(t * FPS) + 1
            if not gt_boxes.get(f):
                continue
            img = cv2.imread(str(d / f"k_{n + 1:04d}.jpg"))
            for tid, b in gt_boxes[f]:
                feat, frac = color_feature(torso_crop(img, b), bg)
                samples.append((n, team_of.get(tid), tid, feat, frac))
    else:
        from ultralytics import YOLO

        model = YOLO(a.model)
        step = max(1, len(index) // a.det_frames)
        for n, t in index[::step][: a.det_frames]:
            f = round(t * FPS) + 1
            img = cv2.imread(str(d / f"k_{n + 1:04d}.jpg"))
            r = model.predict(img, imgsz=1920, conf=0.25, classes=[0], verbose=False)[0]
            for x0, y0, x1, y1 in r.boxes.xyxy.tolist():
                b = (x0, y0, x1 - x0, y1 - y0)
                if not on_court(b):
                    continue
                m = max(((iou(b, gb), tid) for tid, gb in gt_boxes.get(f, [])), default=(0, None))
                tid = m[1] if m[0] >= 0.5 else None
                feat, frac = color_feature(torso_crop(img, b), bg)
                samples.append((n, team_of.get(tid) if tid else None, tid, feat, frac))

    model = fit_team_model([s[3] for s in samples if s[3] is not None and s[4] >= 0.15])
    preds = [model.predict(s[3], s[4]) for s in samples]

    players = [(s[1], p) for s, p in zip(samples, preds) if s[1]]
    acc, swapped = best_mapping(players)
    fix = (lambda p: {"A": "B", "B": "A"}.get(p, p)) if swapped else (lambda p: p)
    res = {
        "video": a.video, "mode": a.mode, "keyframes": len({s[0] for s in samples}),
        "player_detections": len(players),
        "coverage": round(sum(p != "?" for _, p in players) / max(1, len(players)), 3),
        "accuracy_on_assigned": round(acc, 3),
    }
    by_track = defaultdict(list)
    for s, p in zip(samples, preds):
        if s[2] is not None:
            by_track[s[2]].append(fix(p))
    res["track_votes"] = {str(t): {"gt": team_of.get(t), "pred": vote(v), "n": len(v)} for t, v in sorted(by_track.items())}
    res["track_accuracy"] = round(np.mean([team_of.get(t) == vote(v) for t, v in by_track.items()]), 3)
    if a.mode == "det":
        others = [p for s, p in zip(samples, preds) if s[1] is None]
        res["non_player_on_court_detections"] = len(others)
        res["non_player_assigned_a_team"] = round(sum(p != "?" for p in others) / max(1, len(others)), 3)
    res["court_mask"] = not a.no_court_mask
    print(json.dumps(res, indent=1))
    out = Path("data/_teams") / f"{a.video}_{a.mode}{'_nomask' if a.no_court_mask else ''}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
