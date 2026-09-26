"""Evaluate the team of each shot (shooter's track -> colour clustering) against labels.

  uv run python scripts/eval_shot_teams.py IMG_0104 ...
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.court import get_court
from hoopstats.detection.runner import load_detections
from hoopstats.evaluation import match_events
from hoopstats.events import detect_shots
from hoopstats.events.make_model import MakeModel
from hoopstats.events.shooter import find_shooter
from hoopstats.outputs import read_events_csv
from hoopstats.schema import ShotEvent, Team, Zone
from hoopstats.teams.assign import fit_video_team_model, shot_team, track_teams

FPS = 30000 / 1001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args()
    mm = MakeModel.load("models/make_model.json")
    tot_ok = tot = tot_unknown = 0
    for v in a.videos:
        cache = Path(a.cache_dir) / v
        tr = dict(np.load(cache / "ball_track.npz"))
        _, persons = load_detections(cache)
        cal = Calibration.load(f"calib/{v}.json")
        court = get_court(cal.court_standard)
        teams, stats = track_teams(f"data/outdoor/{v}.MOV", persons, cal, court, cache)
        model, bg = fit_video_team_model(f"data/outdoor/{v}.MOV", persons, cal, court, cache)
        cap = cv2.VideoCapture(f"data/outdoor/{v}.MOV")
        rr = abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2
        shots = detect_shots(tr, cal.rim_center, rr, FPS, make_model=mm)
        pred = [ShotEvent(i, s.release_frame / FPS, s.release_frame, Team.UNKNOWN, Zone.INSIDE_ARC, s.made)
                for i, s in enumerate(shots)]
        gt = read_events_csv(f"labels/{v}.csv")
        pairs = []
        for i, j in match_events(gt, pred, 1.0):
            tid, _ = find_shooter(persons, tr, shots[j].release_frame)
            team = "?"
            if tid is not None:
                team, _ = shot_team(cap, persons, tid, shots[j].release_frame, model, bg)
                if team == "?":
                    team = teams.get(tid, "?")          # fall back to the track's keyframe vote
            pairs.append((gt[i].team.value, team, gt[i].timestamp_s))
        same = sum(g == p for g, p, _ in pairs)
        swap = sum(p != "?" and g != p for g, p, _ in pairs)
        ok, unk = max(same, swap), sum(p == "?" for _, p, _ in pairs)
        flip = swap > same
        wrong = [round(t, 1) for g, p, t in pairs if p != "?" and ((g == p) == flip)]
        print(f"{v}: team correct {ok}/{len(pairs)}  unknown {unk}  ({stats['tracks']} tracks, "
              f"{stats['detections']} keyframe detections)  wrong at {wrong}  unknown at "
              f"{[round(t, 1) for g, p, t in pairs if p == '?']}")
        tot_ok, tot, tot_unknown = tot_ok + ok, tot + len(pairs), tot_unknown + unk
        counts = collections.Counter(teams.values())
        print("   track votes:", dict(counts))
    print(f"TOTAL team accuracy {tot_ok}/{tot} = {tot_ok / tot:.1%} (unknown {tot_unknown})")


if __name__ == "__main__":
    main()
