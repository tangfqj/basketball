# Milestone 1 — Report

| | |
|---|---|
| **Date** | 2026-09-27 |
| **Scope** | Fixed-camera 3x3 video (TrackID3x3 Outdoor): shot attempts, make/miss, zone (inside / beyond the arc / free throw), team, points |
| **Code** | `hoopstats analyze <video> --out <dir> [--render]` — see README |

## What the system does

```
video ──► detection (ball: half-res frame + 4K rim window; players + ByteTrack)      [GPU]
      ──► ball tracking (one ball per frame, gaps filled)
      ──► shots: upward flight from below the rim that peaks above it; release time
      ──► make/miss: learned model on the rim-crossing (position, net braking, fall depth, …)
      ──► shooter (closest upper body at release) ──► floor point at release ──► zone / free throw
      ──► team: bib-colour clustering on the shooter's un-occluded views (+ track vote fallback)
      ──► events.csv, stats.json, run_log.json, annotated.mp4
```

Everything is derived from pixels; no scoreboard or on-screen text is used.

## Results vs. requirements (§7.3 targets)

Five labelled development videos IMG_0104–0108 (174 shots), full pipeline on the Mac detections:

| Metric | Target | Result | Kind of number |
|---|---|---|---|
| Attempt recall | ≥ 90% | **174/174 (100%)** | 0105–0108 were never used to tune shot detection |
| Attempt precision | ≥ 90% | **97.8%** | extra detections are real shots after the whistle or rim-level near misses |
| Make/miss | ≥ 95% | **97.1%**; 95.6% leave-one-video-out; **97.4% on IMG_0108** (untouched until the final test) | IMG_0108 is the honest number |
| Zone | ≥ 90% | **97.7%** (free throws 9/9) | development score |
| Team | ≥ 90% | **89.7%** — just below | development score |
| Speed | ≤ 3× real time | A100 (Colab): 11.3 fps ≈ 3.1× real time; Mac M2: 4.6 fps ≈ 6.5× | measured |

Cloud vs. Mac detections: ball positions differ by a median of 1.3 px; accuracy differences are within a
few borderline shots (Colab run, before the team-fallback fix: precision 96.1%, make/miss 96.0%, zone 96.6%).

Components: ball detector `ball_v1` precision 0.977 / recall 0.971 on held-out frames; ball tracking recall
96–97%; every labelled shot's flight is tracked to the rim (174/174).

Details: `doc/experiments/` (one note per step, including negative results).

## Known limitations

- **Team (89.7%)**: in crowded shots the contesting defender is sometimes taken as the shooter. Ranking by
  ball possession was tried and made it worse. Next idea: player detection on every frame (now every 3rd).
- **No held-out test**: the planned test on IMG_0111–0115 (requirements EV-5) was dropped by decision
  (2026-09-27). All zone/team numbers above are development scores; only make/miss has an untouched-video
  result.
- **Indoor subset** not evaluated (20 fps, tiny hoop; test-only by decision).
- Shots after the final whistle are counted (no game-state awareness); fouled shots count as attempts (SH-5).
- Release time is the detected start of the shooting motion + 0.4 s (measured offset); on 0105–0108 the
  labels were pre-filled from the detector, so release accuracy there is not independently measured.
- Speed is limited by 4K HEVC decoding on the CPU; `--workers N` runs parallel detection processes
  (identical results; tested on CPU, not yet timed on a GPU).
- Only one court/camera (TrackID3x3 Outdoor); a new setup needs calibration (`hoopstats calibrate`, or
  `scripts/derive_calibration.py` for the same camera slightly moved).

## How to run

Local (Mac): `uv sync --extra ml`, then `uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render`.
Cloud: `uv run python scripts/make_colab_bundle.py`, upload the bundle and videos to Drive, run
`notebooks/run_pipeline_colab.ipynb`. Accuracy table: `uv run python scripts/eval_outputs.py IMG_0104 … --out-dir out`.
