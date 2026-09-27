# Basketball Shot Analytics — Development Plan (Milestone 2: per-player shots)

| | |
|---|---|
| **Status** | Approved v0.1 |
| **Related** | [requirements.md §11](requirements.md), [plan.md](plan.md) (Milestone 1) |
| **Last updated** | 2026-09-27 |

## Status

| Step | Status |
|---|---|
| 0 Bib legibility probe | **Done — go.** Numbers legible front and back; see `doc/experiments/2026-09-27-bib-probe.md` |
| 1 Shooter ground truth | **Done.** 174 shots reviewed (19 proposals corrected) |
| 2 Shooter accuracy | **89.1%** with the M1 rule; learned ranking tried, worse (85.6%); see `2026-09-27-shooter-accuracy.md` |
| 3 Bib reader | Crops + training notebook ready (`notebooks/train_bib_reader.ipynb`); waiting for the Colab run |
| 4 Identity assignment + pipeline | – |
| 5 Held-out test + report | – |

## Idea

Per-player counting = **who shot** (shooter attribution, from M1) × **who that is** (bib number).

The bib problem is simpler than general number recognition:

- **Closed set.** Each video has 6 players, 3 per team, and the team is already known from colour.
  The reader only has to choose among the 3 numbers of that team, or say "unreadable".
- **Tracks, not frames.** One confident reading anywhere along a track labels the whole track,
  so the number does not have to be legible at the moment of the shot.
- **Free labels.** The dataset gives every player box in every frame, with track ID → bib number.
  Thousands of labeled torso crops are available without hand labeling.

## Method

1. **Shooter** — M1 rule (player whose upper body is nearest the ball just after release).
   Measured against reviewed ground truth; improved if it is the weak link.
2. **Bib reader** — small image classifier on torso crops: classes = bib numbers seen in the dev
   videos + `unreadable`. Trained on crops from dev videos (labels from the dataset). An
   off-the-shelf OCR model is tried first as a zero-training baseline.
3. **Roster** — the 3 numbers per team in a video, from the most confident readings.
4. **Track → number** — confidence-weighted vote over the track's readings, restricted to the
   team's roster. Tracks without a confident reading inherit from the most likely preceding /
   following track of the same team (position and time continuity); otherwise `#?`.
5. **Shot → player** — the shooter track's number; if unknown, read the bib in a window around
   the release.
6. **Outputs** — `players.csv`, `player` column in `events.csv`, `players` in `stats.json`.

## Steps

| Step | Work | Deliverable | Kevin |
|---|---|---|---|
| 0 | Crop dataset boxes from 4K frames; check front/back numbers, size, fraction legible; try OCR. **Go / no-go** for bib-based identity. | `doc/experiments/bib-probe.md` | Look at crops |
| 1 | Pre-fill shooter track + number for the 174 dev shots from dataset boxes; review tool. | `labels/*.csv` with `shooter_id`, `shooter_number` | Review (~20–30 min) |
| 2 | Shooter accuracy of the M1 rule; fix the main error modes. | Experiment note | – |
| 3 | Crop dataset builder; train bib classifier (Colab); crop-level and track-level metrics. | `models/bib_v1`, notebook | Run notebook |
| 4 | Roster, track → number, shot → player; wire into `hoopstats analyze` and `evaluate`. | Pipeline + dev results | – |
| 5 | Label and run the held-out test (IMG_0111–0115, also closes the M1 test); report. | `doc/milestone2-report.md` | Label test shots |

## Fallback

If step 0 shows bibs are too often unreadable, per-player IDs fall back to anonymous per-video
identities (A1–A3, B1–B3) with a thumbnail per player; to be decided with Kevin.
