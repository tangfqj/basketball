"""Pre-fill the shooter of every labeled shot, for review with `hoopstats shooter-review` (M2 step 1).

For each shot in labels/<video>.csv, the dataset player boxes (TrackID3x3 MOT, every frame) are ranked
with the Milestone 1 rule (upper body nearest the cached ball track just after the labeled release).
The proposal is only a starting point: every shot is reviewed by hand.

Writes data/shooter_frames/<video>/items.json and one JPEG crop per shot and offset (git-ignored).
Decisions are stored by the review tool in labels/shooters/<video>.json.

  uv run python scripts/propose_shooters.py IMG_0104 IMG_0105 ...   # resumable (--budget-s)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from hoopstats.datasets.trackid3x3 import load_delimitation, load_mot, team_membership
from hoopstats.events.shooter import _near_release
from hoopstats.outputs import read_events_csv

GT_ROOT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
OUT = Path("data/shooter_frames")
FPS = 30000 / 1001
OFFSETS = (-15, -8, 0, 6)          # frames relative to the labeled release (-0.5 s .. +0.2 s)
CROP_W, CROP_H = 1600, 1000         # 4K pixels
IMG_W, IMG_H = 3840, 2160


def gt_persons(video: str) -> np.ndarray:
    """(N, 7) [frame (0-based), id, x, y, w, h, 1] from the dataset MOT file (1-based frames)."""
    return np.array([(d.frame - 1, d.track_id, d.x, d.y, d.w, d.h, 1.0)
                     for d in load_mot(GT_ROOT / "MOT" / f"{video}.txt")], dtype=float)


def propose(persons, track, release):
    near = _near_release(persons, track, release, person_every=1, look_ahead=9)
    ranked = sorted(near.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
    return [[int(t), round(d, 3), n] for t, (d, n) in ranked]


def crop_window(persons, track, release, tid):
    pts = []
    if tid is not None:
        s = persons[(persons[:, 1] == tid) & (persons[:, 0] == release)]
        if len(s):
            _, _, x, y, w, h, _ = s[0]
            pts.append((x + w / 2, y + h / 2))
    for f in range(release, release + 10):
        if f < len(track["x"]) and np.isfinite(track["x"][f]):
            pts.append((track["x"][f], track["y"][f]))
            break
    cx, cy = np.mean(pts, axis=0) if pts else (IMG_W / 2, IMG_H / 2)
    x0 = int(np.clip(cx - CROP_W / 2, 0, IMG_W - CROP_W)) // 2 * 2
    y0 = int(np.clip(cy - CROP_H / 2, 0, IMG_H - CROP_H)) // 2 * 2
    return x0, y0


def extract(video: Path, release: int, x0: int, y0: int, out_dir: Path, stem: str) -> None:
    first = max(0, release + OFFSETS[0])
    n = [release + o - first for o in OFFSETS]
    sel = "+".join(f"eq(n\\,{k})" for k in n)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{first / FPS:.4f}", "-i", str(video),
                    "-vf", f"select='{sel}',crop={CROP_W}:{CROP_H}:{x0}:{y0}", "-vsync", "0",
                    "-frames:v", str(len(n)), "-q:v", "3", str(out_dir / f"{stem}_%d.jpg")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--budget-s", type=float, default=0, help="stop extracting after this many seconds (resume later)")
    a = ap.parse_args()
    t_start = time.time()
    for v in a.videos:
        out = OUT / v
        out.mkdir(parents=True, exist_ok=True)
        events = load_delimitation(GT_ROOT / "delimitation_frames" / f"{v}.csv")
        jersey = next(e.jersey for e in events if e.jersey)
        team = team_membership(events)
        persons = gt_persons(v)
        track = dict(np.load(Path(a.cache_dir) / v / "ball_track.npz"))
        items = []
        for ev in read_events_csv(f"labels/{v}.csv"):
            r = ev.frame if ev.frame >= 0 else round(ev.timestamp_s * FPS)
            cands = propose(persons, track, r)
            tid = cands[0][0] if cands else None
            x0, y0 = crop_window(persons, track, r, tid)
            stem = f"e{ev.event_id:03d}"
            images = []
            for k, o in enumerate(OFFSETS, start=1):
                f = r + o
                boxes = [{"id": int(p[1]), "x": round(p[2] - x0, 1), "y": round(p[3] - y0, 1),
                          "w": round(p[4], 1), "h": round(p[5], 1)} for p in persons[persons[:, 0] == f]]
                ball = None
                if 0 <= f < len(track["x"]) and np.isfinite(track["x"][f]):
                    ball = [round(track["x"][f] - x0, 1), round(track["y"][f] - y0, 1)]
                images.append({"name": f"{stem}_{k}.jpg", "offset": o, "boxes": boxes, "ball": ball})
            items.append({"event_id": ev.event_id, "frame": r, "t": round(ev.timestamp_s, 3),
                          "label_team": ev.team.value, "zone": ev.zone.value, "made": ev.made,
                          "crop": [x0, y0, CROP_W, CROP_H], "proposal": tid, "candidates": cands[:4],
                          "images": images})
        players = {str(t): {"number": jersey.get(t), "team": team.get(t)} for t in sorted(set(jersey) | set(team))}
        (out / "items.json").write_text(json.dumps({"video": v, "players": players, "items": items}, indent=1))

        done = skipped = 0
        for it in items:
            stem = f"e{it['event_id']:03d}"
            if all((out / im["name"]).exists() for im in it["images"]):
                continue
            if a.budget_s and time.time() - t_start > a.budget_s:
                skipped += 1
                continue
            extract(Path("data/outdoor") / f"{v}.MOV", it["frame"], it["crop"][0], it["crop"][1], out, stem)
            done += 1
        no_ball = sum(it["proposal"] is None for it in items)
        print(f"{v}: {len(items)} shots, {no_ball} without proposal; extracted {done}, remaining {skipped}")


if __name__ == "__main__":
    main()
