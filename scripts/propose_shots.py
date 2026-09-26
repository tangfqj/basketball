"""Pre-fill shot labels from the detector, for review in the labeling tool (requirements EV-4).

Writes labels/<video>.csv with one row per detected shot: the release time (shifted by the typical
detector offset), the predicted make/miss, zone "inside_arc" as a placeholder, no team, and the note
"auto" — every "auto" row counts as incomplete until you confirm it with C in the labeling tool.

  uv run python scripts/propose_shots.py IMG_0105          # refuses to overwrite existing labels
  uv run hoopstats label data/outdoor/IMG_0105.MOV         # then review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoopstats.events import detect_shots
from hoopstats.events.make_model import MakeModel
from hoopstats.labeling.server import write_labels

FPS = 30000 / 1001
RELEASE_SHIFT_S = 0.40   # detector releases are ~0.4 s early on IMG_0104 (doc/experiments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--make-model", default="models/make_model.json", help="'' = rule v1")
    ap.add_argument("--force", action="store_true", help="overwrite existing labels")
    ap.add_argument("--out-dir", default="labels")
    ap.add_argument("--blank-made", action="store_true",
                    help="leave made/missed empty (for test videos: avoids anchoring the labels to the model)")
    a = ap.parse_args()
    for v in a.videos:
        out = Path(a.out_dir) / f"{v}.csv"
        if out.exists() and not a.force:
            print(f"{v}: {out} exists — not overwriting (use --force)")
            continue
        tr = dict(np.load(Path(a.cache_dir) / v / "ball_track.npz"))
        cal = json.loads(Path(f"calib/{v}.json").read_text())
        rr = abs(cal["rim_edge"][1][0] - cal["rim_edge"][0][0]) / 2
        mm = MakeModel.load(a.make_model) if a.make_model and Path(a.make_model).exists() else None
        shots = detect_shots(tr, cal["rim_center"], rr, FPS, make_model=mm)
        rows = []
        for s in shots:
            t = s.release_frame / FPS + RELEASE_SHIFT_S
            rows.append({"timestamp_s": f"{t:.3f}", "frame": round(t * FPS), "team": "", "zone": "inside_arc",
                         "made": "" if a.blank_made else ("1" if s.made else "0"), "note": "auto"})
        write_labels(out, rows)
        print(f"{v}: {len(rows)} proposed shots ({sum(s.made for s in shots)} made) -> {out}")


if __name__ == "__main__":
    main()
