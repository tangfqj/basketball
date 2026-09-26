# Experiment: make/miss on new videos — rule v1 vs. learned model (Phase 3.2)

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Data** | 136 labelled shots, IMG_0104–0107 (49 made); 0105–0107 labelled in review mode after rule v1 was frozen |
| **Held out** | IMG_0108 is not used at all — final test |

## Shot attempts on videos the rules were not tuned on (0105–0107)

Recall **103 / 103**, precision 103 / 105. Release times: the labels were created from proposals shifted by
+0.4 s and mostly not moved, so the release offset on these videos is **not** an independent measurement.

## Make / miss

| | 0104 | 0105 | 0106 | 0107 | Total |
|---|---|---|---|---|---|
| Rule v1 (tuned on 0104) | 33/33 | 29/33 | 24/35 | 32/35 | 118/136 (86.8%) |
| Logistic regression, leave-one-video-out | 32/33 | 31/33 | 34/35 | 33/35 | **130/136 (95.6%)** |

Rule v1 failed in two systematic ways:
- **Low shots / layups** (apex < 60 px above the rim): the ball is not tracked through the net, so the
  "reaches net depth under the rim" test fails → makes called misses.
- **Fast, flat long shots** crossing near the rim centre but **not braked** (horizontal speed kept,
  vertical speed *increasing*: 17–24 → 24–34 px/frame): balls falling past the front/back of the rim —
  the head-on depth ambiguity → misses called makes.

The model (`src/hoopstats/events/make_model.py`, `models/make_model.json`, L2 = 1; results are stable for
L2 = 0.3–10) combines: |dx| at the crossing, change in vertical speed (net braking), horizontal braking,
apex height, fall depth after the crossing, bounce, under-rim flag, apparent size. Largest weights:
|dx| (−2.6), under the rim (+1.2), fall depth (−0.9), vertical braking (−0.9).

## Next

- Final check on **IMG_0108** with labels made *without* the model's make/miss suggestions
  (`propose_shots.py --blank-made`).
- Remaining errors (6) to inspect visually; a net-motion cue (pixel change in the net) is the next candidate
  feature if the depth ambiguity dominates.

## Final test: IMG_0108 (never used for development; made/missed labelled without suggestions)

Model frozen before the test (`models/make_model.json`, sha1 bc2fe68a…, commit 706ff40).

| | Result |
|---|---|
| Attempts | recall **38 / 38**, precision 38 / 39 |
| Make / miss — learned model | **37 / 38 (97.4%)** |
| Make / miss — rule v1 (comparison) | 29 / 38 (76.3%) |

- The single error (362.7 s): a miss predicted as made with probability 0.60 — crossing 0.5 rim radii from the
  centre, little braking; close to the decision boundary.
- The extra detection (299.7 s) crosses the rim level 5 rim radii to the side and is classified as a miss,
  so it would add an attempt but no points.
- Release times on IMG_0105–0108 come from shifted proposals and are therefore not an independent measurement.

**Phase 3 closed.** Requirements targets: attempt recall ≥ 90% ✓ (100%), precision ≥ 90% ✓ (97–100%),
make/miss ≥ 95% ✓ (95.6% leave-one-video-out, 97.4% on the untouched video).

## Known weakness to watch: "no ball" frames

Across IMG_0104 and IMG_0108 the tracker reports a ball on 8 of 19 frames labelled "no ball" (mostly
bridging a briefly hidden ball, or a low-confidence detection). It has not caused shot errors so far, but it
matters for Phase 4 (shooter = player holding the ball) and should be re-checked there.
