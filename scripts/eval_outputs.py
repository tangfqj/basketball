"""Summarise pipeline outputs (out/<video>/) against the labels: accuracy per video + speed.

  uv run python scripts/eval_outputs.py IMG_0104 IMG_0105 ... [--out-dir out] [--markdown]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hoopstats.evaluation import evaluate
from hoopstats.outputs import read_events_csv


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--rules", default="3x3")
    a = ap.parse_args()
    rows, tot = [], {"gt": 0, "pred": 0, "match": 0, "made": 0, "zone": 0, "team": 0, "team_n": 0, "vid_s": 0.0, "proc_s": 0.0}
    for v in a.videos:
        d = Path(a.out_dir) / v
        if not (d / "events.csv").exists():
            print(f"{v}: no outputs in {d}")
            continue
        log = json.loads((d / "run_log.json").read_text())
        det = log.get("detection") or {}
        speed = det.get("fps")
        if not Path(f"labels/{v}.csv").exists():
            rows.append((v, "no labels", "", "", "", "", "", f"{speed}"))
            continue
        gt, pred = read_events_csv(f"labels/{v}.csv"), read_events_csv(d / "events.csv")
        r = evaluate(gt, pred, a.rules)
        rows.append((v, f"{r.n_matched}/{r.n_gt}", pct(r.attempt_precision), pct(r.make_accuracy), pct(r.zone_accuracy),
                     pct(r.team_accuracy), f"{r.points_error.get('total')}", f"{speed} fps" if speed else "cached"))
        tot["gt"] += r.n_gt
        tot["pred"] += r.n_pred
        tot["match"] += r.n_matched
        tot["made"] += round((r.make_accuracy or 0) * r.n_matched)
        tot["zone"] += round((r.zone_accuracy or 0) * r.n_matched)
        n_team = sum(1 for i, _ in r.matches if gt[i].team.value != "?")
        tot["team"] += round((r.team_accuracy or 0) * n_team)
        tot["team_n"] += n_team
        tot["vid_s"] += log.get("video_s", 0)
        tot["proc_s"] += log.get("processing_s", 0)
    hdr = ("video", "shots found", "precision", "make/miss", "zone", "team", "points err", "detection speed")
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if tot["gt"]:
        m = max(1, tot["match"])
        print(f"| **total** | {tot['match']}/{tot['gt']} | {pct(tot['match'] / max(1, tot['pred']))} | {pct(tot['made'] / m)} | "
              f"{pct(tot['zone'] / m)} | {pct(tot['team'] / max(1, tot['team_n']))} | | "
              f"{tot['proc_s'] / 60:.1f} min for {tot['vid_s'] / 60:.1f} min of video |")


if __name__ == "__main__":
    main()
