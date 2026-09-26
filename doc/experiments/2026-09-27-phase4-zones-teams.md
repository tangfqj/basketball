# Experiment: shooter, zone and team per shot (Phases 4–5)

| | |
|---|---|
| **Date** | 2026-09-27 |
| **Data** | 174 labelled shots, IMG_0104–0108 (zones and teams from the shot labels) |
| **Code** | `events/shooter.py`, `teams/assign.py`, `scripts/eval_zones.py`, `scripts/eval_shot_teams.py` |

All numbers below are **development scores**: the methods were adjusted after looking at errors on these
videos. The held-out test videos (IMG_0111–0115) give the final numbers.

## Shooter and zone

- Shooter = player whose upper body is closest to the ball during the first frames of the flight.
- Shooting position = shooter's floor point **at the detected release** (start of the shooting motion,
  feet planted). First version used the lowest box bottom in the 0.5 s before release: 87.4% — it picks the
  point farthest from the basket, e.g. the start of a drive.
- Free throw = shooter at the free-throw spot **and** free-throw context: moved < 0.4 m in the 2 s before,
  nobody within 2.5 m. Observed: free throws < 0.2 m / ≥ 3.1 m; in-play shots from the spot 3–5 m / 1.4–2.6 m.

| Label → predicted | inside | beyond | FT |
|---|---|---|---|
| inside (81) | **78** | 3 | 0 |
| beyond (84) | 1 | **83** | 0 |
| FT (9) | 0 | 0 | **9** |

**Zone accuracy 170 / 174 (97.7%)**. All 4 errors are within ~1.5 m of the 3-point line.

Label correction: IMG_0105 54.9 s was labelled `inside_arc` but is a free throw (TrackID3x3 notes a referee
release at 50.9 s and "FT success" at 56.1 s; every other referee-release event is followed by a labelled FT).

## Team per shot

| Method | 0104 | 0105 | 0106 | 0107 | 0108 | Total |
|---|---|---|---|---|---|---|
| Track vote at keyframes | 24/33 | 27/33 | – | – | – | 77% (0104–0105) |
| Shooter's own box before release | 30/33 | 29/33 | – | – | – | 89% (0104–0105) |
| + only views without overlap (−1.5 s … +0.5 s) | 31/33 | 30/33 | 33/35 | 28/35 | 34/38 | **156/174 (89.7%)** |

- ByteTrack at 10 fps fragments players into 150–190 track pieces per 6-min video, so track-level votes
  are unreliable; classifying the shooter's own box around the release works better.
- Remaining errors: crowded drives (defender overlapping or chosen as shooter). IMG_0107 (purple vs yellow)
  is the weakest video (80%) — to investigate.
- Target (≥ 90%) is not yet met on the development videos.

## Follow-up: why teams fail on IMG_0107 (80%)

- The colour model is **not** the problem: on the dataset's player boxes it assigns 96.8% of IMG_0107
  detections correctly (0106: 98.2%, 0108: 95.1%).
- The errors are **wrong shooters**: in crowded shots the "closest upper body just after release" is often
  the contesting defender (the ball is already travelling towards the contesting hand).
- **Negative result — possession before release:** ranking players by how often the ball was inside their
  box before the release made things worse (zones 97.7% → 93.1–96.0% for windows of 0.2–1 s; teams on
  IMG_0107 80% → 74%): the defender's box also contains the ball in crowded frames, and longer windows
  reach back to the passer. Kept as an option (`use_possession`, off).
- Fragility: several candidate players often tie at distance 0; tie-breaking (prefer the player seen in
  more frames, deterministic order) changes results by 1–2 shots.
- Most promising next step: person detection on every frame (now every 3rd) for fewer track fragments
  and cleaner boxes in crowds — a speed trade-off to decide in Phase 6.
