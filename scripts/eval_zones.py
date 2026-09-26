"""Evaluate shooter position / zone (Phase 4) on labelled videos.

  uv run python scripts/eval_zones.py IMG_0104 IMG_0105 ...
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.court import get_court, is_beyond_arc
from hoopstats.detection.runner import load_detections
from hoopstats.evaluation import match_events
from hoopstats.events import detect_shots
from hoopstats.events.make_model import MakeModel
from hoopstats.events.shooter import shooter_info
from hoopstats.outputs import read_events_csv
from hoopstats.schema import ShotEvent, Team, Zone

FPS = 30000 / 1001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--dump", default="cache/zone_eval.json")
    a = ap.parse_args()
    mm = MakeModel.load("models/make_model.json")
    conf = collections.Counter()
    dump = []
    for v in a.videos:
        tr = dict(np.load(Path(a.cache_dir) / v / "ball_track.npz"))
        _, persons = load_detections(Path(a.cache_dir) / v)
        cal = Calibration.load(f"calib/{v}.json")
        court = get_court(cal.court_standard)
        rr = abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2
        shots = detect_shots(tr, cal.rim_center, rr, FPS, make_model=mm)
        pred = [ShotEvent(i, s.release_frame / FPS, s.release_frame, Team.UNKNOWN, Zone.INSIDE_ARC, s.made)
                for i, s in enumerate(shots)]
        gt = read_events_csv(f"labels/{v}.csv")
        for i, j in match_events(gt, pred, 1.0):
            info = shooter_info(persons, tr, shots[j].release_frame, cal, court, FPS)
            z = info.zone or "none"
            conf[(gt[i].zone.value, z)] += 1
            dump.append({"video": v, "t": round(gt[i].timestamp_s, 2), "gt_zone": gt[i].zone.value, "pred_zone": z,
                         "court_xy": info.court_xy, "tid": info.track_id, "score": info.score,
                         "d_arc": None if info.court_xy is None else _arc_dist(court, *info.court_xy)})
    print("confusion (label -> predicted):")
    for (g, p), n in sorted(conf.items()):
        print(f"  {g:10s} -> {p:13s} {n}")
    ok = sum(n for (g, p), n in conf.items() if g == p)
    tot = sum(conf.values())
    print(f"zone accuracy: {ok}/{tot} = {ok / tot:.1%}")
    Path(a.dump).write_text(json.dumps(dump, indent=1))


def _arc_dist(court, x, y):
    """Signed distance to the 3-point line (m): > 0 beyond the arc."""
    d_arc = np.hypot(x, y - court.hoop_y) - court.arc_radius if y > court.arc_break_y else abs(x) - court.corner_x
    return round(float(d_arc), 2) if is_beyond_arc(court, x, y) == (d_arc > 0) else round(float(d_arc), 2)


if __name__ == "__main__":
    main()
