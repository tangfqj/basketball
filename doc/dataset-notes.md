# TrackID3x3 — Dataset Inspection Notes

| | |
|---|---|
| **Inspected** | 2026-09-23 |
| **Local copy** | `data/outdoor/` (2 of 12 videos), `data/indoor/` (all 42 clips); videos only, no annotations yet |

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

## To do

- Download the annotations (player boxes, keypoints, court keypoints, identities) for both subsets.
- Count shots per video / clip to size the dev and test sets.
- Decide on requirement changes (IN-5 frame rate for Indoor, head-on view in IN-4, Indoor's role).
