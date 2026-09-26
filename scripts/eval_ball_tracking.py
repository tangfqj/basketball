"""Evaluate ball tracking (Phase 2 exit criterion) on videos processed with `hoopstats detect` + `track-ball`.

1. Reviewed ball boxes (labels/ball/<video>.json):
   - "ball" frames: hit if the tracked position lies within max(20 px, 0.75 x ball size) of the box centre
   - "none" frames: false alarm if the tracker reports a ball
   Reported separately for train and val frames (val = time block never used for training).
2. Shot labels (labels/<video>.csv), when present: for each shot, the share of frames from release to
   +1.5 s with a ball position, and whether the path comes within 3 rim radii of the rim.

  uv run python scripts/eval_ball_tracking.py IMG_0104 [--cache-dir cache]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args()
    summary = {}
    for v in a.videos:
        tr = np.load(Path(a.cache_dir) / v / "ball_track.npz")
        frames = {f"{v}_f{r['frame']:05d}.jpg": r for r in json.loads(Path(f"data/ball_frames/{v}/frames.json").read_text())}
        labels = json.loads(Path(f"labels/ball/{v}.json").read_text())
        res = {s: {"ball": 0, "hit": 0, "none": 0, "false_alarm": 0, "errors": []} for s in ("train", "val")}
        for name, lab in labels.items():
            r = frames.get(name)
            if r is None or lab["status"] == "skip":
                continue
            f, s = r["frame"], res[r["split"]]
            has = tr["state"][f] > 0
            if lab["status"] == "ball":
                x, y, w, h = lab["box"]
                s["ball"] += 1
                if has:
                    err = float(np.hypot(tr["x"][f] - (x + w / 2), tr["y"][f] - (y + h / 2)))
                    s["errors"].append(err)
                    s["hit"] += err <= max(20.0, 0.75 * (w + h) / 2)
            else:
                s["none"] += 1
                s["false_alarm"] += bool(has)
        out = {}
        for split, s in res.items():
            out[split] = {"ball_frames": s["ball"], "recall": round(s["hit"] / max(1, s["ball"]), 3),
                          "median_error_px": round(float(np.median(s["errors"])), 1) if s["errors"] else None,
                          "none_frames": s["none"],
                          "false_alarm_rate": round(s["false_alarm"] / max(1, s["none"]), 3)}
        shots_p = Path(f"labels/{v}.csv")
        if shots_p.exists():
            rim = json.loads(Path(f"calib/{v}.json").read_text())
            rc, rr = np.array(rim["rim_center"]), abs(rim["rim_edge"][1][0] - rim["rim_edge"][0][0]) / 2
            fps = 30000 / 1001
            cov, reach = [], []
            with open(shots_p, newline="") as fh:
                for row in csv.DictReader(fh):
                    f0 = round(float(row["timestamp_s"]) * fps)
                    win = np.arange(f0, min(f0 + round(1.5 * fps), len(tr["state"])))
                    ok = tr["state"][win] > 0
                    cov.append(ok.mean())
                    d = np.hypot(tr["x"][win][ok] - rc[0], tr["y"][win][ok] - rc[1]) if ok.any() else np.array([np.inf])
                    reach.append(bool(d.min() <= 3 * rr))
            out["shots"] = {"n": len(cov), "median_coverage": round(float(np.median(cov)), 3),
                            "coverage_ge_80pct": int(sum(c >= 0.8 for c in cov)), "path_reaches_rim": int(sum(reach))}
        summary[v] = out
        print(v, json.dumps(out, indent=1))
    Path(a.cache_dir, "ball_tracking_eval.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
