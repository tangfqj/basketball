# Milestone 2, step 4 — player per shot, end to end (dev videos)

`hoopstats analyze` now reads the shooter's bib (`players/bib.py`) from the frames it already decodes
for the team decision, builds a 3-number roster per team and votes the number within the shooter's
team roster. Outputs: `player` + `shooter_track` in events.csv, `players.csv`, `players` in stats.json,
shooter box + `A #10` label in the annotated video.

Evaluation: `scripts/eval_players.py` (labels/shooters, reviewed). For honest numbers each video uses
the bib model that did **not** train on it (`data/bib_results/bib_held_<video>.pt`). Milestone 1 metrics
are unchanged (attempts 174/174, precision 97.8%, make/miss 97.1%, zone 97.7%, team 89.7%).

## Result

| Video | Player (team + number) | Team | Number |
|---|---|---|---|
| IMG_0104 | 28/33 | 30/33 | 28/33 |
| IMG_0105 | 25/33 | 30/33 | 25/33 |
| IMG_0106 | 30/35 | 31/35 | 30/35 |
| IMG_0107 | 28/35 | 28/35 | 31/35 |
| IMG_0108 | 34/38 | 34/38 | 37/38 |
| **Total** | **145/174 = 83.3%** (target 85%) | 87.9% | 86.8% |

Per-player counts: mean |attempts error| 1.2, |makes error| 0.9 per player (28 players, 5 videos).
Rosters (3 numbers per team) were right in all 5 videos.

## Where the errors come from (first run, 143/174)

| Shooter (step 2) | Player right | Team wrong | Number wrong |
|---|---|---|---|
| right (155) | 143 | 8 | 4 |
| wrong (19) | – | 13 | 6 |

Two thirds of the player errors are **wrong shooter** (mostly the contesting defender). With the
right shooter, the player is right 92% of the time; the rest are colour-team errors and a few bib
reads from views that show another player.

## Tuning (offline, `scripts/player_identity_experiment.py`)

Per-shot crop probabilities are saved in `cache/<video>/bib_probs.npz`, so rules can be compared
without re-running the pipeline:

| Views that vote (frames around the detected release) | Team from colour | Team also from bib |
|---|---|---|
| −45…+15 (the team views, first run) | 82.2% | 83.3% |
| **−12…+15 (adopted)** | 83.9% | 84.5% |
| −45…0 | 78.2% | 77.6% |

- Views close to the actual release (the detected release is ~0.4 s early) are the reliable ones:
  earlier views often belong to another player after a track switch. Adopted: −12…+15.
- Letting the bib also decide the team (when its number is in only one roster) gains 1–2 shots out of
  174 — within noise. Not adopted; the approved rule (vote within the colour team's roster) stays.
- The remaining gap to 85% is shooter attribution, not bib reading.
