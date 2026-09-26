"""Evaluate Phase 3 (shot attempts + make/miss) against hand labels, with an error list.

Uses the cached ball track (cache/<video>/ball_track.npz). Zone and team are Phases 4–5, so only
attempt recall/precision and make/miss accuracy are meaningful here.

  uv run python scripts/eval_shots.py IMG_0104 [--max-frame 900] [--track path/to/ball_track.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoopstats.evaluation import evaluate
from hoopstats.events import detect_shots
from hoopstats.outputs import read_events_csv, write_events_csv
from hoopstats.schema import ShotEvent, Team, Zone

FPS = 30000 / 1001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--track", help="ball_track.npz (default: cache/<video>/ball_track.npz)")
    ap.add_argument("--max-frame", type=int, default=None, help="only frames processed so far")
    a = ap.parse_args()
    tr = dict(np.load(a.track or Path(a.cache_dir) / a.video / "ball_track.npz"))
    n = a.max_frame or len(tr["state"])
    if a.max_frame:
        tr = {k: v[:n] for k, v in tr.items()}
    cal = json.loads(Path(f"calib/{a.video}.json").read_text())
    rr = abs(cal["rim_edge"][1][0] - cal["rim_edge"][0][0]) / 2
    shots = detect_shots(tr, cal["rim_center"], rr, FPS)

    pred = [ShotEvent(i, s.release_frame / FPS, s.release_frame, Team.UNKNOWN, Zone.INSIDE_ARC, s.made,
                      confidence=s.confidence) for i, s in enumerate(shots)]
    out = Path(a.cache_dir) / a.video / "events_phase3.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_events_csv(pred, out)
    gt = [e for e in read_events_csv(f"labels/{a.video}.csv") if e.timestamp_s * FPS < n - 45]
    r = evaluate(gt, pred, "3x3")
    print(f"{a.video} (frames < {n}): GT {r.n_gt}  predicted {r.n_pred}  matched {r.n_matched}")
    print(f"  attempt recall {r.attempt_recall:.3f}  precision {r.attempt_precision:.3f}  "
          f"make/miss accuracy {r.make_accuracy:.3f}")
    mg = {i for i, _ in r.matches}
    mp = {j: i for i, j in r.matches}
    offsets = [pred[j].timestamp_s - gt[i].timestamp_s for i, j in r.matches]
    if offsets:
        print(f"  release time offset (pred - label): median {np.median(offsets):+.2f} s, "
              f"range {min(offsets):+.2f}..{max(offsets):+.2f}")
    print("  missed shots (label time, made):", [(round(g.timestamp_s, 2), g.made) for i, g in enumerate(gt) if i not in mg])
    print("  extra detections:")
    for j, (p, s) in enumerate(zip(pred, shots)):
        if j not in mp:
            print(f"    {p.timestamp_s:7.2f} s made={p.made} {s.features}")
    print("  make/miss errors:")
    for i, j in r.matches:
        if gt[i].made != pred[j].made:
            print(f"    {gt[i].timestamp_s:7.2f} s label made={gt[i].made} -> pred {pred[j].made} {shots[j].features}")
    print("  events written to", out)


if __name__ == "__main__":
    main()
