# Basketball Shot Analytics — Development Plan (Milestone 1)

| | |
|---|---|
| **Status** | Draft v0.1 |
| **Related** | [requirements.md](requirements.md) |
| **Last updated** | 2026-09-23 |

## Guiding principles

- **Evaluation first.** Labeled dev clips and a scoring script exist before any detection work, so every
  later step is measured against the metrics in requirements §7.2.
- **Early vertical slice.** Reach an end-to-end "attempts + makes" result (Phase 3) as early as possible;
  ball detection/tracking is the main technical risk and should surface early.
- **Cached stages.** Each pipeline stage writes its intermediate results to disk so downstream stages can be
  iterated without re-running detection.
- **Tune on dev, report on test.** Test clips are held out until the final evaluation (requirements EV-5).

Pipeline: `calibrate → detect → track → events → outputs`

## Phase 0 — Foundations

| Step | Work | Deliverable |
|---|---|---|
| 0.1 | Project skeleton: Python project, dependency management (e.g. `uv`), OpenCV, ffmpeg; module layout per pipeline stage; stage caching. | Runnable package, empty stages |
| 0.2 | Collect ~8–10 candidate videos; verify each against requirements §3 (especially static camera); split into dev and test sets. | `data/` manifest (videos not in git) |
| 0.3 | Keyboard-driven event labeling tool (requirements EV-3). | Labeling tool |
| 0.4 | Label 1–2 dev clips. | Ground-truth `events.csv` files |
| 0.5 | Evaluation script: ±1.0 s one-to-one matching, all §7.2 metrics. | `evaluate.py` |

**Exit criteria:** labeled dev clips exist and an events file can be scored automatically.

## Phase 1 — Calibration

| Step | Work | Deliverable |
|---|---|---|
| 1.1 | Click-on-frame calibration tool: rim points, ≥ 4 court landmarks, court standard (requirements §4). | Calibration tool |
| 1.2 | Compute the image → court homography; re-project court lines onto the frame for visual verification; save `calib.json`. | `calib.json` per video |

**Exit criteria:** re-projected 3-point arc and free-throw line align with the painted lines on all dev clips.

## Phase 2 — Detection & tracking

| Step | Work | Deliverable |
|---|---|---|
| 2.1 | Player detection with a pretrained person detector; sanity-check on dev clips. | Player detections |
| 2.2 | Ball detection: baseline an off-the-shelf model; fine-tune on public basketball datasets; add own labeled frames if needed (requirements TD-3/TD-4). Measure recall specifically in the region around the rim. | Ball detector + recall report |
| 2.3 | Ball tracking: link detections into trajectories, interpolate short gaps with a motion model, reject false positives (heads, shoes, etc.). | Ball trajectories |

**Exit criteria:** the ball is tracked reliably in the frames around labeled shot events.

## Phase 3 — Vertical slice: attempts & makes ★ key checkpoint

| Step | Work | Deliverable |
|---|---|---|
| 3.1 | Shot detection: a trajectory that rises and then heads into the rim region is an attempt; backtrack along it to estimate the release moment. | Attempt events |
| 3.2 | Make/miss: a make is the ball passing from above the rim to below it through the rim's circle; everything else is a miss. | Make/miss per attempt |

**Exit criteria:** end-to-end run on full dev videos with first attempt-recall/precision and make/miss
accuracy numbers. If ball tracking is insufficient, revisit Phase 2 before continuing.

## Phase 4 — Shot type

| Step | Work | Deliverable |
|---|---|---|
| 4.1 | Shooter identification: the player closest to the ball at release. | Shooter per attempt |
| 4.2 | Foot position: bottom of the shooter's bounding box (or pose-estimated feet), mapped to court coordinates; classify 2PT/3PT (requirements TY-2/TY-3). | `court_x, court_y`, shot type |
| 4.3 | Free-throw detection: shooter at the FT line, preceding dead ball, other players lined up along the lane (requirements TY-1). | FT classification |

**Exit criteria:** shot-type accuracy measured on dev clips.

## Phase 5 — Team split

| Step | Work | Deliverable |
|---|---|---|
| 5.1 | Jersey color features from torso crops; cluster into two teams; treat outliers (referees) separately (requirements TM-2/TM-4). | Team per player detection |
| 5.2 | Attribute each attempt to the shooter's team; optional team-name mapping config (TM-3). | Team per attempt |

**Exit criteria:** team accuracy measured on dev clips.

## Phase 6 — Outputs & hardening

| Step | Work | Deliverable |
|---|---|---|
| 6.1 | CLI `analyze <video> --calib <calib.json> --out <dir> [--render]`; `stats.json`, `events.csv`, run log, annotated video (requirements §6). | CLI + outputs |
| 6.2 | Label the test clips; final evaluation on the test set; error analysis. | Evaluation report |
| 6.3 | Performance against NF-2 (≤ 3× video duration): batched inference, optional Core ML export, run heavy models only near candidate shots. | Performance report |
| 6.4 | README and demo clip. | Docs |

**Exit criteria:** all Milestone 1 requirements met or deviations documented.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Small / blurry / occluded ball is detected poorly | Attempts and makes missed | Fine-tuning, trajectory interpolation, 60 fps footage, early checkpoint in Phase 3 |
| Truly static footage is scarce online | Not enough test data | Self-recorded sessions with a phone on a tripod |
| Feet occluded or mid-jump at release | 2PT/3PT errors near the line | Use last ground contact before release; pose estimation |
| Similar jersey colors or lighting changes | Team errors | Input requirement IN-6; per-clip clustering |
| Model licensing | Constraints on code reuse | Decide before Phase 2 (see below) |

## Decisions needed before Phase 2

- **Detector framework / license.** Ultralytics YOLO is AGPL-3.0: acceptable for a personal demo, but if the
  code may later be closed-source or commercial, choose an Apache-licensed alternative (e.g. YOLOX or an
  Apache-licensed RT-DETR implementation) from the start.
- **Target Mac hardware** (requirements open question 1), which decides whether training runs locally or on a cloud GPU.
