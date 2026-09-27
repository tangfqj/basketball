"""Experiment (M2 step 4): should the bib also decide the team?

Re-assigns players offline from cache/<video>/bib_probs.npz (written by `hoopstats analyze`) and the
events of --out, with two rules, and scores both like scripts/eval_players.py:
  team      number voted among the colour team's roster (default of step 4)
  identity  number voted among both rosters; a number found in only one roster also sets the team

  cd scripts && uv run python player_identity_experiment.py IMG_0104 ... --out out/m2   (imports eval_players)
Also varies which views vote: frames around the detected release and the overlap limit.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from eval_players import evaluate

from hoopstats.outputs import read_events_csv, write_events_csv
from hoopstats.players.bib import assign_identities, assign_numbers, select_views
from hoopstats.schema import Team
from hoopstats.training.bib_eval import class_number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="out/m2")
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    grid = [(m, b, af, o) for m in ("team", "identity") for b, af in ((45, 15), (24, 15), (12, 15), (45, 0))
            for o in (0.15, 0.3, 1.0)]
    for mode, before, after, max_ov in grid:
        tot = ok = team_ok = 0
        for v in a.videos:
            z = np.load(Path(a.cache_dir) / v / "bib_probs.npz")
            events = read_events_csv(Path(a.out) / v / "events.csv")
            probs = []
            for i in range(len(events)):
                sel = z["shot"] == i
                m = select_views(z["frame"][sel], z["overlap"][sel], int(z["release"][i]), before, after, max_ov)
                probs.append(z["probs"][sel][m])
            numbers = [class_number(c) for c in z["classes"]]
            teams = [e.team.value for e in events]
            if mode == "team":
                nums, _ = assign_numbers(probs, teams, numbers)
            else:
                teams, nums, _ = assign_identities(probs, teams, numbers)
            for e, t, n in zip(events, teams, nums):
                e.team, e.player = Team(t), n
            (tmp / v).mkdir(exist_ok=True)
            write_events_csv(events, tmp / v / "events.csv")
            rows, _, _ = evaluate(v, str(tmp))
            tot += len(rows)
            ok += sum(r["ok"] for r in rows)
            team_ok += sum(r["team_ok"] for r in rows)
        print(f"{mode:8s} window -{before}..+{after} overlap<={max_ov}: player {ok}/{tot} = {ok / tot:.1%}   "
              f"team {team_ok / tot:.1%}")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
