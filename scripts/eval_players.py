"""Per-player evaluation of pipeline outputs (Milestone 2, requirements §11.3).

Ground truth: labels/<video>.csv (shots) + labels/shooters/<video>.json (shooter: dataset team + bib).
Predictions: <out>/<video>/events.csv from `hoopstats analyze` (columns team, player).
Team letters are arbitrary per video: the predicted letter is mapped to the dataset's team by majority
over the matched shots.

  uv run python scripts/eval_players.py IMG_0104 ... --out out/m2 [--errors]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from hoopstats.evaluation import match_events
from hoopstats.outputs import read_events_csv


def evaluate(video: str, out_dir: str):
    gt = read_events_csv(f"labels/{video}.csv")
    lab = json.loads(Path(f"labels/shooters/{video}.json").read_text())
    pred = read_events_csv(Path(out_dir) / video / "events.csv")
    pairs = match_events(gt, pred, 1.0)
    votes = collections.Counter()
    for i, j in pairs:
        s = lab.get(str(gt[i].event_id), {})
        if s.get("status") == "ok" and pred[j].team.value in "AB":
            votes[(pred[j].team.value, s["team"])] += 1
    same = votes[("A", "A")] + votes[("B", "B")]
    swap = votes[("A", "B")] + votes[("B", "A")]
    tmap = {"A": "A", "B": "B"} if same >= swap else {"A": "B", "B": "A"}
    rows = []
    for i, j in pairs:
        s = lab.get(str(gt[i].event_id), {})
        if s.get("status") != "ok":
            continue
        want = (s["team"], str(s["number"]))
        got = (tmap.get(pred[j].team.value, "?"), pred[j].player or "?")
        rows.append({"video": video, "t": round(gt[i].timestamp_s, 1), "want": want, "got": got,
                     "team_ok": want[0] == got[0], "number_ok": want[1] == got[1], "ok": want == got})
    # per-player counts: all GT shots vs all predicted shots
    gt_counts = collections.defaultdict(lambda: [0, 0])
    for e in gt:
        s = lab.get(str(e.event_id), {})
        if s.get("status") == "ok":
            c = gt_counts[(s["team"], str(s["number"]))]
            c[0] += 1
            c[1] += int(e.made)
    pr_counts = collections.defaultdict(lambda: [0, 0])
    for e in pred:
        c = pr_counts[(tmap.get(e.team.value, "?"), e.player or "?")]
        c[0] += 1
        c[1] += int(e.made)
    return rows, gt_counts, pr_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="out/m2")
    ap.add_argument("--errors", action="store_true")
    a = ap.parse_args()
    allrows, abs_err, n_players = [], [0, 0], 0
    for v in a.videos:
        rows, g, p = evaluate(v, a.out)
        allrows += rows
        n = len(rows)
        ok = sum(r["ok"] for r in rows)
        print(f"{v}: player {ok}/{n} = {ok / n:.1%}  (team {sum(r['team_ok'] for r in rows)}/{n}, "
              f"number {sum(r['number_ok'] for r in rows)}/{n})")
        print("   player      GT att/made   pred att/made")
        for k in sorted(set(g) | set(p), key=lambda k: (k[0], k[1] == "?", int(k[1]) if k[1].isdigit() else 0)):
            ga, gm = g.get(k, (0, 0))
            pa, pm = p.get(k, (0, 0))
            print(f"   {k[0]} #{k[1]:<4}    {ga:>3} / {gm:<3}      {pa:>3} / {pm:<3}")
            if k in g:
                abs_err[0] += abs(ga - pa)
                abs_err[1] += abs(gm - pm)
                n_players += 1
    n = len(allrows)
    ok = sum(r["ok"] for r in allrows)
    print(f"TOTAL player accuracy {ok}/{n} = {ok / n:.1%}; team {sum(r['team_ok'] for r in allrows) / n:.1%}, "
          f"number {sum(r['number_ok'] for r in allrows) / n:.1%}; mean |attempts error| per player "
          f"{abs_err[0] / n_players:.2f}, |makes error| {abs_err[1] / n_players:.2f} ({n_players} players)")
    if a.errors:
        for r in allrows:
            if not r["ok"]:
                print(r)


if __name__ == "__main__":
    main()
