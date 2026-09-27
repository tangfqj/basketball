"""Experiment (M2 step 2): can a learned ranking of shooter candidates beat the M1 nearest-upper-body rule?

Candidates = detected player tracks near the ball after the detected release. Features per candidate,
conditional-logit model (softmax over the candidates of one shot), leave-one-video-out.

  cd scripts && uv run python shooter_ranker_experiment.py IMG_0104 ... IMG_0108   (imports eval_shooters)
Result on IMG_0104-0108: 85.6% vs 89.1% for the M1 rule -> not adopted (doc/experiments/2026-09-27-shooter-accuracy.md).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from eval_shooters import GT_ROOT, gt_player_of_track

from hoopstats.calibration import Calibration
from hoopstats.datasets.trackid3x3 import load_mot
from hoopstats.detection.runner import load_detections
from hoopstats.evaluation import match_events
from hoopstats.events import detect_shots
from hoopstats.events.make_model import MakeModel
from hoopstats.events.shooter import _near_release, find_shooter
from hoopstats.outputs import read_events_csv
from hoopstats.pipeline import RELEASE_SHIFT_S
from hoopstats.schema import ShotEvent, Team, Zone

FPS = 30000 / 1001
FEATS = ["d_near", "d_late", "d_min", "poss_short", "poss_long", "ball_above", "x_off", "jump", "rel_h", "rank_near"]


def ball_points(tr, balls, f):
    pts = []
    if 0 <= f < len(tr["x"]) and np.isfinite(tr["x"][f]):
        pts.append((tr["x"][f], tr["y"][f]))
    if balls is not None:
        sel = balls[(balls[:, 0] == f) & (balls[:, 5] >= 0.15)]
        pts += [(b[1] + b[3] / 2, b[2] + b[4] / 2) for b in sel]
    return pts


def features(persons, tr, balls, r):
    near9 = _near_release(persons, tr, r, 3, 9)
    wide = _near_release(persons, tr, r, 3, 18)
    cands = sorted(wide, key=lambda t: wide[t][0])[:5]
    if not cands:
        return [], np.zeros((0, len(FEATS)))
    rows = []
    hs = {}
    for t in cands:
        s = persons[(persons[:, 1] == t) & (np.abs(persons[:, 0] - r - 6) <= 12)]
        hs[t] = float(np.median(s[:, 5])) if len(s) else np.nan
    hmax = np.nanmax(list(hs.values()))
    order = sorted(cands, key=lambda t: near9.get(t, (9.0, 0))[0])
    for t in cands:
        tb = persons[persons[:, 1] == t]
        def box(f, tb=tb):
            s = tb[np.abs(tb[:, 0] - f) <= 1]
            return s[0] if len(s) else None
        d_late, d_all, above, xoff = [], [], [], []
        for f in range(r, r + 19):
            b = box(f)
            if b is None or not np.isfinite(tr["x"][f]):
                continue
            _, _, x, y, w, h, _ = b
            bx, by = tr["x"][f], tr["y"][f]
            dx = max(0.0, abs(bx - (x + w / 2)) - w / 2)
            d = np.hypot(dx, max(0.0, by - (y + 0.45 * h)) + max(0.0, (y - 0.4 * h) - by)) / h
            d_all.append(d)
            if f >= r + 9:
                d_late.append(d)
            if f <= r + 12:
                above.append((y - by) / h)
                xoff.append(abs(bx - (x + w / 2)) / w)

        def poss(f0, f1):
            n = k = 0
            for f in range(f0, f1 + 1):
                b = box(f)
                if b is None or f % 3:
                    continue
                _, _, x, y, w, h, _ = b
                n += 1
                k += any(x - 0.1 * w <= px <= x + 1.1 * w and y - 0.1 * h <= py <= y + 0.8 * h
                         for px, py in ball_points(tr, balls, f))
            return k / n if n else 0.0

        b0, bottoms = box(r - 6), [box(f) for f in range(r, r + 16)]
        bottoms = [b[3] + b[5] for b in bottoms if b is not None]
        jump = ((b0[3] + b0[5]) - min(bottoms)) / b0[5] if b0 is not None and bottoms else 0.0
        rows.append([
            near9.get(t, (3.0, 0))[0], np.median(d_late) if d_late else 3.0, min(d_all) if d_all else 3.0,
            poss(r - 15, r + 3), poss(r - 45, r - 15), np.mean(above) if above else -1.0,
            np.mean(xoff) if xoff else 2.0, jump, hs[t] / hmax if np.isfinite(hs[t]) else 1.0,
            float(order.index(t) == 0),
        ])
    return cands, np.array(rows, float)


def fit(groups, l2=1.0, iters=3000, lr=0.1):
    X = np.concatenate([g[0] for g in groups])
    mean, std = X.mean(0), X.std(0) + 1e-6
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        grad = l2 * w
        for Xg, y in groups:
            z = ((Xg - mean) / std) @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            grad += ((Xg - mean) / std).T @ (p - y)
        w -= lr * grad / len(groups)
    return mean, std, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    mm = MakeModel.load("models/make_model.json")
    data = []   # (video, cands, X, y or None, baseline_ok)
    for v in a.videos:
        cache = Path("cache") / v
        tr = dict(np.load(cache / "ball_track.npz"))
        balls, persons = load_detections(cache)
        cal = Calibration.load(f"calib/{v}.json")
        rr = abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2
        shots = detect_shots(tr, cal.rim_center, rr, FPS, make_model=mm)
        sh = round(RELEASE_SHIFT_S * FPS)
        pred = [ShotEvent(i, (s.release_frame + sh) / FPS, 0, Team.UNKNOWN, Zone.INSIDE_ARC, s.made)
                for i, s in enumerate(shots)]
        gt = read_events_csv(f"labels/{v}.csv")
        labels = json.loads(Path(f"labels/shooters/{v}.json").read_text())
        gtb = collections.defaultdict(list)
        for d in load_mot(GT_ROOT / "MOT" / f"{v}.txt"):
            gtb[d.frame].append((d.track_id, (d.x, d.y, d.w, d.h)))
        for i, j in match_events(gt, pred, 1.0):
            lab = labels.get(str(gt[i].event_id), {})
            want = lab.get("track") if lab.get("status") == "ok" else None
            r = shots[j].release_frame
            cands, X = features(persons, tr, balls, r)
            ids = [gt_player_of_track(persons, gtb, t, r - 3, r + 12) for t in cands]
            y = np.array([float(g == want) for g in ids])
            base, _ = find_shooter(persons, tr, r, balls=balls)
            base_ok = base is not None and gt_player_of_track(persons, gtb, base, r - 3, r + 12) == want
            data.append((v, cands, X, y if y.sum() == 1 else None, base_ok, gt[i].event_id))
    n = len(data)
    print(f"shots {n}; GT shooter among candidates: {sum(d[3] is not None for d in data)}; "
          f"baseline correct {sum(d[4] for d in data)}")
    tot = 0
    for held in a.videos:
        train = [(d[2], d[3]) for d in data if d[0] != held and d[3] is not None]
        mean, std, w = fit(train, l2=a.l2)
        ok = base = m = 0
        for d in data:
            if d[0] != held:
                continue
            m += 1
            base += d[4]
            if d[3] is not None and len(d[2]):
                ok += d[3][int(np.argmax(((d[2] - mean) / std) @ w))] == 1
        tot += ok
        print(f"{held}: ranker {ok}/{m}  baseline {base}/{m}")
    print(f"TOTAL ranker {tot}/{n} = {tot / n:.1%}   baseline {sum(d[4] for d in data)}/{n}")
    mean, std, w = fit([(d[2], d[3]) for d in data if d[3] is not None], l2=a.l2)
    print("weights (standardised):", {f: round(x, 2) for f, x in zip(FEATS, w)})


if __name__ == "__main__":
    main()
