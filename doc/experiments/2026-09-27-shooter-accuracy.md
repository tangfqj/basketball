# Milestone 2, step 2 — how often is the shooter right?

Ground truth: `labels/shooters/IMG_0104–0108.json`, 174 shots, reviewed by hand (155 proposals kept,
19 corrected). Script: `scripts/eval_shooters.py` — the pipeline's shooter track (detected players,
ByteTrack) is mapped to a dataset player by box overlap (IoU ≥ 0.5, majority over the frames after
the release) and compared with the label.

## M1 rule (nearest upper body just after the detected release)

| Video | Correct |
|---|---|
| IMG_0104 | 30/33 |
| IMG_0105 | 28/33 |
| IMG_0106 | 30/35 |
| IMG_0107 | 32/35 |
| IMG_0108 | 35/38 |
| **Total** | **155/174 = 89.1%** (18 wrong player, 1 box not matched) |

- 13 of the 18 wrong players are on the **other team**: the contesting defender. This is the same
  error behind the 89.7% team accuracy of Milestone 1.
- Most errors have distance 0 for two players: the ball is inside both the shooter's and the
  defender's box (crowded shots under the basket, overlapping boxes). Box geometry alone cannot
  separate them.

## Tried: learned ranking of candidates (negative result)

`scripts/shooter_ranker_experiment.py`: up to 5 candidate tracks per shot, 10 box/ball features
(distances in two windows, ball-in-box before the release, ball above the head, horizontal offset,
jump, box height, rank), softmax over the candidates, leave-one-video-out.

| | Correct |
|---|---|
| M1 rule | 155/174 (89.1%) |
| Ranker (L2 = 1, 10, 50) | 149/174 (85.6%) |

Not adopted. The features describe boxes, and in the failure cases both boxes look alike.

## Implication for the targets

Player accuracy ≈ shooter accuracy × bib accuracy. With 89.1% shooter accuracy, the 85% target
leaves room for ~4–5% bib errors. A cue that could separate shooter and defender is **hand
position** (pose keypoints: which wrists hold the ball as it leaves), to be tried only if needed.
