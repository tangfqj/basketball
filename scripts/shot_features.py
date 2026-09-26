"""Dump detector features of all labelled shots (matched to labels) to a CSV for analysis / training.

  uv run python scripts/shot_features.py IMG_0104 IMG_0105 ... -> cache/shot_features.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hoopstats.evaluation import match_events
from hoopstats.events import detect_shots
from hoopstats.outputs import read_events_csv
from hoopstats.schema import ShotEvent, Team, Zone

FPS = 30000 / 1001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="cache/shot_features.csv")
    a = ap.parse_args()
    rows = []
    for v in a.videos:
        tr = dict(np.load(Path(a.cache_dir) / v / "ball_track.npz"))
        cal = json.loads(Path(f"calib/{v}.json").read_text())
        rr = abs(cal["rim_edge"][1][0] - cal["rim_edge"][0][0]) / 2
        shots = detect_shots(tr, cal["rim_center"], rr, FPS)
        pred = [ShotEvent(i, s.release_frame / FPS, s.release_frame, Team.UNKNOWN, Zone.INSIDE_ARC, s.made)
                for i, s in enumerate(shots)]
        gt = read_events_csv(f"labels/{v}.csv")
        for i, j in match_events(gt, pred, 1.0):
            rows.append({"video": v, "t": round(gt[i].timestamp_s, 2), "label_made": int(gt[i].made),
                         "label_zone": gt[i].zone.value, "pred_made": int(shots[j].made),
                         **{k: (round(x, 3) if isinstance(x, float) else x) for k, x in shots[j].features.items()}})
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("video", "t", "label_made", "pred_made"), k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "shots ->", a.out)


if __name__ == "__main__":
    main()
