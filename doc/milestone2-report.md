# Milestone 2 — Report

| | |
|---|---|
| **Status** | Wrapped up 2026-09-27 (final held-out test not run) |
| **Related** | [requirements.md §11](requirements.md), [plan-milestone2.md](plan-milestone2.md), [milestone1-report.md](milestone1-report.md) |

## What the system does

Milestone 2 adds **per-player** shot counts to Milestone 1. For every shot, the shooter's bib number is
read from the video, so each shot is credited to a player identified by **team + bib number**
(e.g. `A #10`) within one video.

1. **Shooter** — Milestone 1 rule: the player whose upper body is nearest the ball just after release.
2. **Bib reader** — a small image classifier (YOLO11n-cls, 128 px torso crops) trained on crops cut from
   the TrackID3x3 player boxes, whose bib numbers come with the dataset (no hand labeling).
3. **Roster** — per video, the 3 most-read numbers of each team (3x3: three players per team).
4. **Player per shot** — confidence-weighted vote over the shooter's crops around the release,
   restricted to the team's roster; `#?` if no crop is confident.

Outputs of `hoopstats analyze`: `players.csv` (team, number, attempts, makes, FG%, points), a `player`
column in `events.csv`, a `players` section in `stats.json`, and in `annotated.mp4` a box around the
shooter labeled with team + number and a banner such as `A #11  2PT  MADE` (3x3 labels: 1PT / 2PT).

## Results (dev videos IMG_0104–0108, 174 shots)

| Metric | Result | Target |
|---|---|---|
| Player per shot (team + number) | **83.3%** (145/174) | ≥ 85% — **not met** |
| Shooter (diagnostic) | 89.1% | reported |
| Bib, per crop (leave-one-video-out) | 99.3% | reported |
| Bib, per 3-s window with roster | 99.85% | ≥ 90% |
| Per-player count error | 1.2 attempts, 0.9 makes per player | reported |
| Milestone 1 metrics | unchanged (attempts 100%, precision 97.8%, make/miss 97.1%, zone 97.7%, team 89.7%) | |

Each video was scored with a bib model that did not see it in training. Details:
`doc/experiments/2026-09-27-{bib-probe,shooter-accuracy,bib-reader-v1,players-dev}.md`.

## Known limitations

- **Wrong shooter is the main error** (2/3 of the player errors): in crowded shots under the basket the
  contesting defender's box contains the ball too. A pose-based rule (whose hands release the ball) is
  the most promising fix; not attempted.
- **Colour-team errors** remain (team 87.9%); the player is only right when the team is.
- **The bib reader partly recognises players**, not only digits: the same people wear the same bibs
  across the dataset's videos. Accuracy on players never seen in training is unknown.
- **Final held-out test (IMG_0111–0115) not run**; it needs shot labels and reviewed shooters for
  those videos (also still open from Milestone 1).
- Identity is per video only; substitutions and players beyond the 3 per team are not handled.

## How to run

Local (Mac): put the bib model at `data/models/bib_v1.pt` (= `bib_final.pt` from the bib notebook), then
`uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render-range 12:32`.

Cloud (Colab): `uv run python scripts/make_colab_bundle.py` (includes the bib model), upload
`data/colab/hoopstats_bundle.zip` to `MyDrive/hoopstats/`, run `notebooks/run_pipeline_colab.ipynb`.

Evaluation: `uv run python scripts/eval_players.py IMG_0104 … --out out` (players),
`uv run python scripts/eval_outputs.py IMG_0104 … --out-dir out` (Milestone 1 metrics).
Bib reader training: `notebooks/train_bib_reader.ipynb`. Shooter labels: `uv run hoopstats shooter-review`.
