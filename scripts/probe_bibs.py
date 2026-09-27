"""Step 0 of Milestone 2: are bib numbers legible? (doc/plan-milestone2.md)

Samples frames of an Outdoor video, crops every dataset player box at native 4K and writes contact
sheets labeled with track id and bib number (from the dataset), for visual inspection.

    python scripts/probe_bibs.py IMG_0104 --frames 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hoopstats.datasets.trackid3x3 import load_delimitation, load_mot, team_membership  # noqa: E402

GT_ROOT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
FPS = 30000 / 1001


def grab(video: Path, frame1: int) -> np.ndarray:
    """Frame by 1-based MOT index (index 1 = first frame), native resolution, BGR."""
    t = (frame1 - 1) / FPS
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", str(video), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def upper_body(img, x, y, w, h, out_h=240):
    """Head-to-hip crop (top 60% of the box), scaled to a fixed height."""
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + 0.6 * h)
    c = img[max(0, y0):y1, max(0, x0):x1]
    s = out_h / max(1, c.shape[0])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), out_h)), (x1 - x0, y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="data/_bibs")
    a = ap.parse_args()

    events = load_delimitation(GT_ROOT / "delimitation_frames" / f"{a.video}.csv")
    jersey = next(e.jersey for e in events if e.jersey)
    team = team_membership(events)
    start = next(e.frame for e in events if "start" in e.label.lower())
    end = next(e.frame for e in events if "end" in e.label.lower())
    boxes = defaultdict(list)
    for d in load_mot(GT_ROOT / "MOT" / f"{a.video}.txt"):
        boxes[d.frame].append(d)

    out = Path(a.out) / a.video
    out.mkdir(parents=True, exist_ok=True)
    frames = np.linspace(start + 30, end - 30, a.frames).astype(int)
    sizes = []
    for f in frames:
        img = grab(Path("data/outdoor") / f"{a.video}.MOV", int(f))
        tiles = []
        for d in sorted(boxes.get(int(f), []), key=lambda d: (team.get(d.track_id, "?"), d.track_id)):
            tile, (cw, ch) = upper_body(img, d.x, d.y, d.w, d.h)
            sizes.append((cw, ch))
            lab = f"{team.get(d.track_id, '?')} #{jersey.get(d.track_id, '?')} ({cw}x{ch})"
            tile = cv2.copyMakeBorder(tile, 28, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, lab, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        if tiles:
            cv2.imwrite(str(out / f"f{int(f):06d}.jpg"), cv2.hconcat(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    w = np.array(sizes)
    print(f"{a.video}: {len(frames)} frames, {len(sizes)} crops; crop px median {np.median(w, 0)}, "
          f"min {w.min(0)}, max {w.max(0)}; jersey {jersey}; teams {team}")


if __name__ == "__main__":
    main()
