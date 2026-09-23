# TrackID3x3 — Dataset Inspection Notes

| | |
|---|---|
| **Inspected** | 2026-09-23 |
| **Local copy** | `data/outdoor/` (2 of 12 videos), `data/indoor/` (all 42 clips), annotations in `data/trackid3x3_repo/` (sparse clone of the GitHub repo) |

## Technical properties

| | Outdoor | Indoor |
|---|---|---|
| Files | `IMG_0104.MOV` (6.3 min), `IMG_0105.MOV` (7.4 min) | 42 clips `basket_S{1-6}T{1-7}_{pre,post}.mp4`, 1.5–15 s each, 6.3 min total |
| Resolution / codec | 3840×2160, HEVC | 1280×720, H.264 |
| Frame rate | 29.97 fps | **~20 fps** (999/50), below requirement IN-5 (≥ 30 fps) |
| Camera stability within a video | Static: < 1 px drift (at 960 px width) over the whole video | Static |
| Camera position across videos | ~2 px shift (at 960 px width) between the two videos → one shared calibration plus a small per-video refinement | Identical across all 42 clips (< 1 px) → one calibration |

## Visual observations

### Outdoor
- **View:** elevated, from the half-court end **facing the basket head-on** (not a side view of the rim).
  The whole half court, 3-point arc, lane, and free-throw line are clearly visible: white lines on a
  blue surface, good for calibration.
- **Hoop:** a portable stand with grass and trees behind it; the rim is small in the frame (≈ 60 px wide at 4K).
- **Make/miss risk:** with a head-on view, a ball passing just in front of or behind the rim can look like a
  make (depth ambiguity). Expect make/miss to need more than a simple "ball crosses rim" rule.
- **Teams:** numbered bibs; color pairs change per game (green vs. purple, pink vs. yellow).
- **Non-players in frame:** waiting teams in bibs beside the court, a referee / scorekeeper, a coach at the
  sideline, spectators. People must be filtered to the court area before team clustering.

### Indoor
- **View:** elevated corner/sideline view of a large gym; the active hoop is on the far wall and small
  (ball ≈ 12–15 px).
- **Court markings:** several overlapping line systems (blue, white, black, yellow); which arc is the
  relevant 3-point line must be confirmed with the dataset's court keypoints.
- **Teams:** mixed bib colors (red, orange, pink vs. black, blue, lime); two-team separation is not obvious (IN-6 borderline).
- **Content:** short structured clips (sessions S1–S6, trials T1–T7, pre/post); may be drills with 0–1 shots each.
- Spectators and a coach stand at the court edge.

## Assessment

- **Outdoor:** good fit as the primary dev/test source. Main risk: make/miss from a head-on view.
- **Indoor:** weak fit for end-to-end shot evaluation (20 fps, tiny hoop, confusing markings, unclear teams,
  few shots). More useful for component metrics (calibration, player detection) and as a stress test.

## Annotations (from the GitHub repo, `ground_truth/`)

| Path | Content | Use for us |
|---|---|---|
| `Outdoor/MOT/<video>.txt` | Player boxes for the 6 on-court players, every frame, MOTChallenge format (`frame, id, x, y, w, h, ...`) in 4K pixels. All 12 videos. | Player detection metrics (Phase 2.1) |
| `Outdoor/delimitation_frames/<video>.csv` | Game-flow events with frame numbers: game start/end, out of bounds, resume, foul, violation, and **some** `FG success` / `FG failure` / `FT success`; plus track id → jersey number and offense/defense track ids. | Team ground truth per track (Phase 5); rough cross-check for shots. **Not complete shot labels**: only 11 FG successes and 13 failures across all 12 videos. |
| `Outdoor/court_keypoints/Outdoor.ndjson` | 4 court-corner keypoints (Labelbox export); court positions in the dataset scripts: 1505 × 1105 cm. | Calibration (CAL-6); verified visually: overlay matches the painted lines. |
| `Outdoor/pose/IMG_0104_.json` | 10 body keypoints, **first 3600 frames of IMG_0104 only**. | Foot localization metric (Phase 4), limited |
| `Outdoor/MOT/split_transformed/{check_ball,free_throw}/` | Court-coordinate tracks split into possession segments, labeled by offense/defense + jersey (e.g. `O10`). | Probably not needed |
| `Indoor/MOT`, `transformed_MOT*`, `coordinates_with_color` | Boxes and court tracks for 42 clips; players labeled **O1red, O2blue, O3pink vs D1black, D2orange, D3yellow**. | Confirms: Indoor teams are **not** separable by color |
| `Indoor/court_keypoints`, `Indoor/pose` | 12 court keypoints; pose for 13 clips | Calibration check |

## Decision (2026-09-23)

- **Outdoor** = main dev/test source.
- **Indoor** = test only, for attempts and make/miss; excluded from team and development/tuning.

## To do

- Count shots per video / clip to size the dev and test sets.
- Decide on requirement changes (IN-5 frame rate for Indoor, head-on view in IN-4, Indoor's role).
