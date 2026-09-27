"""Shooter accuracy of the pipeline against the reviewed shooter labels (Milestone 2 step 2).

For every labeled shot matched by a detected shot (+-1 s), the predicted shooter track (detected players,
ByteTrack) is mapped to a dataset player by box overlap in the frames after the release, and compared with
labels/shooters/<video>.json.

  uv run python scripts/eval_shooters.py IMG_0104 IMG_0105 ... [--errors]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.datasets.trackid3x3 import load_mot
from hoopstats.detection.runner import load_detections
from hoopstats.evaluation import match_events
from hoopstats.events import detect_shots
from hoopstats.events.make_model import MakeModel
from hoopstats.events.shooter import find_shooter
from hoopstats.outputs import read_events_csv
from hoopstats.pipeline import RELEASE_SHIFT_S
from hoopstats.schema import ShotEvent, Team, Zone

GT_ROOT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
FPS = 30000 / 1001


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def gt_player_of_track(persons, gt_by_frame, tid, f0, f1, min_iou=0.5):
    """Dataset track id that the detected track `tid` overlaps most in frames [f0, f1] (majority vote)."""
    votes = collections.Counter()
    for f, _, x, y, w, h, _ in persons[(persons[:, 1] == tid) & (persons[:, 0] >= f0) & (persons[:, 0] <= f1)]:
        best = max(((iou((x, y, w, h), b), g) for g, b in gt_by_frame.get(int(f) + 1, [])), default=(0, None))
        if best[0] >= min_iou:
            votes[best[1]] += 1
    return votes.most_common(1)[0][0] if votes else None


def evaluate(video: str, cache_dir: str, mm) -> list[dict]:
    cache = Path(cache_dir) / video
    tr = dict(np.load(cache / "ball_track.npz"))
    balls, persons = load_detections(cache)
    cal = Calibration.load(f"calib/{video}.json")
    rr = abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2
    shots = detect_shots(tr, cal.rim_center, rr, FPS, make_model=mm)
    shift = round(RELEASE_SHIFT_S * FPS)
    pred = [ShotEvent(i, (s.release_frame + shift) / FPS, s.release_frame + shift, Team.UNKNOWN, Zone.INSIDE_ARC,
                      s.made) for i, s in enumerate(shots)]
    gt = read_events_csv(f"labels/{video}.csv")
    labels = json.loads(Path(f"labels/shooters/{video}.json").read_text())
    gt_by_frame = collections.defaultdict(list)
    for d in load_mot(GT_ROOT / "MOT" / f"{video}.txt"):
        gt_by_frame[d.frame].append((d.track_id, (d.x, d.y, d.w, d.h)))
    rows = []
    for i, j in match_events(gt, pred, 1.0):
        lab = labels.get(str(gt[i].event_id), {})
        r = shots[j].release_frame
        tid, score = find_shooter(persons, tr, r, balls=balls)
        got = gt_player_of_track(persons, gt_by_frame, tid, r - 3, r + 12) if tid is not None else None
        want = lab.get("track") if lab.get("status") == "ok" else None
        if want is None:
            res = "no label"
        elif tid is None:
            res = "no shooter"
        elif got is None:
            res = "box not matched"
        else:
            res = "correct" if got == want else "wrong player"
        rows.append({"video": video, "event_id": gt[i].event_id, "t": round(gt[i].timestamp_s, 1), "result": res,
                     "want": want, "got": got, "pred_track": tid, "score": score})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--errors", action="store_true", help="list every shot that is not correct")
    a = ap.parse_args()
    mm = MakeModel.load("models/make_model.json")
    allrows = []
    for v in a.videos:
        rows = evaluate(v, a.cache_dir, mm)
        c = collections.Counter(r["result"] for r in rows)
        n = len(rows) - c["no label"]
        print(f"{v}: shooter correct {c['correct']}/{n}  {dict(c)}")
        allrows += rows
    c = collections.Counter(r["result"] for r in allrows)
    n = len(allrows) - c["no label"]
    print(f"TOTAL shooter accuracy {c['correct']}/{n} = {c['correct'] / max(1, n):.1%}  {dict(c)}")
    if a.errors:
        for r in allrows:
            if r["result"] != "correct":
                print(r)


if __name__ == "__main__":
    main()
