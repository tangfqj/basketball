"""Torso crops labeled with bib numbers, for training the bib reader (Milestone 2 step 3, PGT-2).

Labels are free: the TrackID3x3 Outdoor ground truth has a box for every on-court player in every frame
(MOT) and the bib number of every track (delimitation_frames). For every `every`-th frame the torso of
each box is cropped at native 4K, letterboxed to a square and saved as JPEG; index.csv records the label
and how much of the torso other players cover (training uses the un-occluded crops).

  hoopstats bib-crops data/outdoor/IMG_0104.MOV --every 8          # -> data/bib_crops/IMG_0104/
"""

from __future__ import annotations

import csv
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..datasets.trackid3x3 import load_delimitation, load_mot, team_membership
from ..video import probe

GT_ROOT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
TORSO = (0.15, 0.55)        # torso band as fractions of the box height (bib digits sit inside it)
CROP_SIZE = 128
MIN_BOX_H = 150             # px (4K): smaller boxes are partial / far away
FIELDS = ["file", "video", "frame", "track", "team", "number", "occl", "box_w", "box_h"]


def torso_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x, y + TORSO[0] * h, w, (TORSO[1] - TORSO[0]) * h


def occlusion(box, others, grid: int = 16) -> float:
    """Fraction of `box` (x, y, w, h) covered by the union of `others`."""
    x, y, w, h = box
    gx = x + (np.arange(grid) + 0.5) * w / grid
    gy = y + (np.arange(grid) + 0.5) * h / grid
    X, Y = np.meshgrid(gx, gy)
    cov = np.zeros_like(X, bool)
    for ox, oy, ow, oh in others:
        cov |= (X >= ox) & (X <= ox + ow) & (Y >= oy) & (Y <= oy + oh)
    return float(cov.mean())


def letterbox(img: np.ndarray, size: int = CROP_SIZE, pad: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), pad, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def crop_torso(frame: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = torso_box(*box)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return letterbox(frame[y0:y1, x0:x1])


def _frames(video: Path, every: int, start: int, end: int | None, width: int, height: int):
    """Yield (0-based frame index, BGR frame) for every `every`-th frame in [start, end)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start / (30000 / 1001):.4f}"]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-frames:v", str(max(0, (end - start + every - 1) // every))]
    cmd += ["-vf", f"select='not(mod(n\\,{every}))'", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3)
    n = width * height * 3
    k = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield start + k * every, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def build_bib_crops(video: str | Path, out_root: str | Path = "data/bib_crops", every: int = 8, start: int = 0,
                    end: int | None = None, gt_root: str | Path = GT_ROOT) -> dict:
    video = Path(video)
    name = video.stem
    gt_root = Path(gt_root)
    events = load_delimitation(gt_root / "delimitation_frames" / f"{name}.csv")
    jersey = next((e.jersey for e in events if e.jersey), {})
    team = team_membership(events)
    boxes = defaultdict(list)            # 0-based frame -> [(track, (x, y, w, h))]
    for d in load_mot(gt_root / "MOT" / f"{name}.txt"):
        boxes[d.frame - 1].append((d.track_id, (d.x, d.y, d.w, d.h)))
    info = probe(video)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    frames = 0
    for f, img in _frames(video, every, start, end, info.width, info.height):
        frames += 1
        here = boxes.get(f, [])
        ids = [t for t, _ in here]
        for tid, b in here:
            if tid not in jersey or b[3] < MIN_BOX_H or ids.count(tid) > 1:   # >1: ambiguous GT (IMG_0105)
                continue
            crop = crop_torso(img, b)
            if crop is None:
                continue
            occ = occlusion(torso_box(*b), [ob for ot, ob in here if ot != tid])
            fn = f"{name}_f{f:06d}_t{tid}.jpg"
            cv2.imwrite(str(out / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": fn, "video": name, "frame": f, "track": tid, "team": team.get(tid, ""),
                         "number": jersey[tid], "occl": round(occ, 3), "box_w": round(b[2]), "box_h": round(b[3])})
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"video": name, "frames": frames, "crops": len(rows), "seconds": round(time.time() - t0, 1),
            "numbers": sorted(set(jersey.values()))}
