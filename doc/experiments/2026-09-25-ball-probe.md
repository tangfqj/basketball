# Experiment: pretrained YOLO on IMG_0104 (Phase 2 feasibility)

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Script** | `scripts/probe_ball_detection.py` (`--step 4 --rim 1825 845`) |
| **Model** | Ultralytics `yolo11s.pt`, COCO-pretrained, no fine-tuning; CPU |
| **Data** | 33 labelled shots of IMG_0104; frames from 0.3 s before to 1.8 s after release, every 4th frame (528 frames) |
| **Raw numbers** | [`2026-09-25-ball-probe-summary.json`](2026-09-25-ball-probe-summary.json) |

## Setups

- **full**: whole frame downscaled 4K → 1920×1080, `imgsz=1920` (ball ≈ 17 px)
- **rim**: 1280×960 crop around the rim at native 4K, `imgsz=1280` (ball ≈ 35 px)

## Results

| Metric | conf ≥ 0.1 | conf ≥ 0.25 | conf ≥ 0.5 |
|---|---|---|---|
| Frames with a ball detection, full frame | 54% | 40% | 21% |
| Frames with a ball detection, rim crop | 51% | 34% | 18% |
| Same, rim crop, ≥ 0.6 s after release | 58% | 35% | 18% |

- Shots with ≥ 1 ball detection (conf ≥ 0.25) near the rim ≥ 0.6 s after release: **31 / 33**.
- Players (on-court only, vs. TrackID3x3 MOT boxes, IoU ≥ 0.5, conf ≥ 0.25): **recall 93%, precision 82%**.
- Ball detections are ≈ 35 px at 4K and cluster around the rim (median offset < 35 px).

## Interpretation

- There is no ball ground truth, so the rates are **detection rates**, not recall: in some frames the
  ball is legitimately hidden (in a player's hands, behind a body, in the net).
- Visual check of sampled frames: detections that fire are almost always the ball (high precision);
  duplicate detections are rare (0.2% of frames).
- A pretrained detector finds the ball in only about one frame in three at a usable confidence. That is
  enough to *seed* tracking and annotation, but **not** enough for make/miss decisions, which depend on the
  few frames when the ball passes the rim. **Fine-tuning the ball detector is required** (plan 2.2).
- The rim crop at native resolution did not beat the downscaled full frame for the pretrained model; re-test
  after fine-tuning (higher resolution should matter more once the model knows what a small ball looks like).
- Player detection is good enough without fine-tuning for now (plan 2.1).
