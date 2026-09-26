"""Leave-one-video-out evaluation and training of the make/miss classifier.

  uv run python scripts/shot_features.py IMG_0104 IMG_0105 IMG_0106 IMG_0107
  uv run python scripts/train_make_model.py [--save models/make_model.json]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from hoopstats.events.make_model import FEATURES, feature_vector, fit


def parse(r):
    out = {}
    for k, v in r.items():
        if v in ("", None):
            out[k] = None
        elif v in ("True", "False"):
            out[k] = v == "True"
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cache/shot_features.csv")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--save", help="train on all videos and save the model here")
    a = ap.parse_args()
    with open(a.features, newline="") as fh:
        rows = [parse(r) for r in csv.DictReader(fh)]
    videos = sorted({r["video"] for r in rows})
    total_ok = total = rule_ok = 0
    for held in videos:
        tr = [r for r in rows if r["video"] != held and feature_vector(r) is not None]
        X = np.stack([feature_vector(r) for r in tr])
        y = np.array([r["label_made"] for r in tr], float)
        m = fit(X, y, l2=a.l2)
        te = [r for r in rows if r["video"] == held]
        ok = 0
        for r in te:
            fv = feature_vector(r)
            pred = bool(m.prob(fv[None])[0] >= 0.5) if fv is not None else False
            ok += pred == bool(r["label_made"])
        rule = sum(bool(r["pred_made"]) == bool(r["label_made"]) for r in te)
        print(f"held out {held}: model {ok}/{len(te)} ({ok / len(te):.1%})   rule v1 {rule}/{len(te)}")
        total_ok, total, rule_ok = total_ok + ok, total + len(te), rule_ok + rule
    print(f"leave-one-video-out: model {total_ok}/{total} ({total_ok / total:.1%})   rule v1 {rule_ok}/{total} ({rule_ok / total:.1%})")
    X = np.stack([feature_vector(r) for r in rows if feature_vector(r) is not None])
    y = np.array([r["label_made"] for r in rows if feature_vector(r) is not None], float)
    m = fit(X, y, l2=a.l2)
    print("weights (standardised):", {f: round(float(w), 2) for f, w in zip(FEATURES, m.w)})
    if a.save:
        m.save(a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
