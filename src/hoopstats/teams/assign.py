"""Team per player track and per shot (Phase 5 wired to Phase 4).

1. Decode only the video's keyframes (~1 per second; fast) at 1920 px width.
2. On each keyframe, take the on-court player detections (nearest person frame), torso colour features.
3. Fit the per-video 2-team colour model on all of them; predict every detection.
4. Team of a track = majority vote of its keyframe detections. The shot's team = its shooter's track team.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..court import is_on_court
from .color import color_feature, estimate_court_hue, fit_team_model, torso_crop, vote

SCALE = 0.5   # keyframes at 1920 px; detections are in 4K pixels


def extract_keyframes(video: str | Path, out_dir: Path) -> list[tuple[int, Path]]:
    """-> [(frame index, jpg path)] for every keyframe; cached in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.txt"
    if not index.exists():
        log = subprocess.run(["ffmpeg", "-hide_banner", "-skip_frame", "nokey", "-i", str(video),
                              "-vf", "scale=1920:-1,showinfo", "-vsync", "0", "-q:v", "3", str(out_dir / "k_%05d.jpg")],
                             capture_output=True, text=True, check=True).stderr
        times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", log)]
        index.write_text("".join(f"{i + 1} {t}\n" for i, t in enumerate(times)))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    items = []
    for line in index.read_text().splitlines():
        n, t = line.split()
        items.append((round(float(t) * fps), out_dir / f"k_{int(n):05d}.jpg"))
    return items


def track_teams(video: str | Path, persons: np.ndarray, calibration, court, cache: Path,
                person_every: int = 3) -> tuple[dict[int, str], dict]:
    """-> ({track id: 'A' | 'B' | '?'}, stats)."""
    keyframes = extract_keyframes(video, cache / "keyframes")
    cal_half = calibration.scaled(int(calibration.image_size[0] * SCALE), int(calibration.image_size[1] * SCALE))
    first = cv2.imread(str(keyframes[0][1]))
    bg = estimate_court_hue(first, cal_half, court)
    samples = []                       # (track id, feature, sat_frac)
    for f, path in keyframes:
        pf = f - f % person_every
        dets = persons[persons[:, 0] == pf]
        if not len(dets):
            continue
        img = None
        for _, tid, x, y, w, h, _ in dets:
            (cx, cy), = calibration.image_to_court([(x + w / 2, y + h)])
            if tid < 0 or not is_on_court(court, cx, cy, margin=0.3):
                continue
            img = cv2.imread(str(path)) if img is None else img
            feat, frac = color_feature(torso_crop(img, (x * SCALE, y * SCALE, w * SCALE, h * SCALE)), bg)
            samples.append((int(tid), feat, frac))
    model = fit_team_model([s[1] for s in samples if s[1] is not None and s[2] >= 0.15])
    per_track = defaultdict(list)
    for tid, feat, frac in samples:
        per_track[tid].append(model.predict(feat, frac))
    teams = {tid: vote(v) for tid, v in per_track.items()}
    return teams, {"keyframes": len(keyframes), "detections": len(samples), "tracks": len(teams), "court_hue": bg}


def fit_video_team_model(video: str | Path, persons: np.ndarray, calibration, court, cache: Path,
                         person_every: int = 3):
    """Colour model of a video fitted on on-court players at the keyframes (+ court hue at 4K scale)."""
    keyframes = extract_keyframes(video, cache / "keyframes")
    cal_half = calibration.scaled(int(calibration.image_size[0] * SCALE), int(calibration.image_size[1] * SCALE))
    bg = estimate_court_hue(cv2.imread(str(keyframes[0][1])), cal_half, court)
    feats = []
    for f, path in keyframes:
        dets = persons[persons[:, 0] == f - f % person_every]
        img = None
        for _, tid, x, y, w, h, _ in dets:
            (cx, cy), = calibration.image_to_court([(x + w / 2, y + h)])
            if not is_on_court(court, cx, cy, margin=0.3):
                continue
            img = cv2.imread(str(path)) if img is None else img
            feat, frac = color_feature(torso_crop(img, (x * SCALE, y * SCALE, w * SCALE, h * SCALE)), bg)
            if feat is not None and frac >= 0.15:
                feats.append(feat)
    return fit_team_model(feats), bg


def _overlap_frac(box, others) -> float:
    """Largest fraction of `box` covered by any other box."""
    x, y, w, h = box
    best = 0.0
    for ox, oy, ow, oh in others:
        iw = max(0.0, min(x + w, ox + ow) - max(x, ox))
        ih = max(0.0, min(y + h, oy + oh) - max(y, oy))
        best = max(best, iw * ih / max(1.0, w * h))
    return best


def shot_team(cap: cv2.VideoCapture, persons: np.ndarray, tid: int, release: int, model, bg,
              before: int = 45, after: int = 15, step: int = 3, max_overlap: float = 0.15,
              on_view=None) -> tuple[str, list[str]]:
    """Team of the shooter from their own box around the release. Only views where the shooter is not
    overlapped by another player vote (a defender in front mixes both bibs into the torso crop); if there
    are none, all views vote. `on_view(frame_index, full_res_frame, box_xywh, overlap)` is called for every
    view (used to read the bib from the same decoded frames)."""
    rows = persons[(persons[:, 1] == tid) & (persons[:, 0] >= release - before) & (persons[:, 0] <= release + after)]
    wanted = {}
    for r in rows:
        f = int(r[0])
        others = persons[(persons[:, 0] == f) & (persons[:, 1] != tid)][:, 2:6]
        wanted[f] = (r, _overlap_frac(r[2:6], others))
    if not wanted:
        return "?", []
    f0 = min(wanted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    clean, all_views = [], []
    for f in range(f0, max(wanted) + 1):
        if not cap.grab():
            break
        if f not in wanted or (f - f0) % step:
            continue
        _, img = cap.retrieve()
        (_, _, x, y, w, h, _), ov = wanted[f]
        if on_view is not None:
            on_view(f, img, (x, y, w, h), ov)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        feat, frac = color_feature(torso_crop(img, (x * SCALE, y * SCALE, w * SCALE, h * SCALE)), bg)
        lab = model.predict(feat, frac)
        all_views.append(lab)
        if ov <= max_overlap:
            clean.append(lab)
    use = clean if any(v != "?" for v in clean) else all_views
    return vote(use), use
