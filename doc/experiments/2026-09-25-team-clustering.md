# Experiment: team clustering by bib colour (Phase 5)

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Code** | `src/hoopstats/teams/color.py`, `scripts/eval_team_clustering.py` |
| **Data** | IMG_0104 (green vs. purple bibs), IMG_0105 (yellow vs. pink); keyframes ≈ 1 fps at 1920 px |
| **Ground truth** | TrackID3x3 player boxes (MOT) + team per track from the offense/defense lists |

## Method

Torso crop (central 50% width, 15–50% height of the person box) → hue/saturation histogram of saturated
pixels, **excluding the court-surface hue** (estimated from random floor points via the calibration) →
k-means (k = 2) per video → reject detections with < 15% coloured pixels or far from both centres →
majority vote per track. No labels are used for fitting.

## Results

| Video | Boxes | Mode | Player dets | Coverage | Accuracy (assigned) | Track accuracy | Non-players given a team |
|---|---|---|---|---|---|---|---|
| IMG_0104 | dataset (all keyframes) | gt | 2133 | 93.0% | 96.8% | 6 / 6 | – |
| IMG_0105 | dataset (all keyframes) | gt | 2647 | 94.8% | 95.8% | 6 / 6 | – |
| IMG_0104 | YOLO, on-court only (60 keyframes) | det | 326 | 97.9% | 98.1% | 6 / 6 | 25% |
| IMG_0105 | YOLO, on-court only (60 keyframes) | det | 320 | 98.1% | 99.4% | 6 / 6 | 25% |

Ablation without the court-hue mask (IMG_0104, det): accuracy 98.1%, but **53%** of non-player
detections got a team. The main non-player on court is a referee/coach in black standing at the court
edge; the blue court behind him was being read as a bib colour.

## Notes

- "Non-players" = on-court detections not matching a dataset player box at IoU ≥ 0.5; a few of these are
  real players with poorly overlapping boxes, so 25% is an upper bound.
- Errors on single detections come mainly from overlapping players (the torso crop contains the other
  player's bib); track-level voting removes them.
- Remaining work when tracks exist (Phase 2): vote per track, and mark tracks that are mostly "?" (referee) as non-players.
