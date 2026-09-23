# Basketball Shot Analytics — Requirements (Milestone 1)

| | |
|---|---|
| **Status** | Draft v0.3 |
| **Owner** | Kevin |
| **Last updated** | 2026-09-23 |

**Changelog**
- v0.3 — Indoor subset restricted to test-only (attempts, make/miss).
- v0.2 — Adopted the TrackID3x3 dataset (Indoor + Outdoor) as the primary data source. Added scoring
  rule sets (5v5 / 3x3); shot types are now reported as court *zones* plus *points*. Added FIBA 3x3 court
  standard, short-clip input support, component-level metrics, and dataset-related open questions.
- v0.1 — Initial draft.

## 1. Purpose

Build a demo system that takes a fixed-camera basketball video and outputs shot statistics
(attempts, makes, misses, split by shot zone and team) **derived purely from visual understanding of
the video**. The project is a demonstration of video understanding, so reading a scoreboard, captions,
or any other overlaid text is explicitly out of scope as a source of truth.

## 2. Scope

### 2.1 In scope (Milestone 1)

- Detect every **shot attempt** and classify it as **made** or **missed**.
- Classify each attempt by **zone**: inside the arc, beyond the arc, or free throw.
- Convert zones to **points** according to a selectable **scoring rule set** (5v5 or 3x3, see §5.5).
- Attribute each attempt to one of **two teams**.
- Produce aggregated statistics and a per-shot event log.
- Offline (batch) processing on a local machine.

### 2.2 Out of scope (Milestone 1)

- Broadcast / TV footage (moving camera, cuts, replays, graphics).
- Moving-camera footage, including drone footage (e.g. the TrackID3x3 Drone subset).
- Scoreboard OCR or any use of on-screen text.
- Per-player statistics, jersey-number recognition, player re-identification.
- Rebounds, assists, steals, blocks, turnovers, fouls.
- Full-court footage with two hoops.
- Real-time / streaming processing.
- Fully automatic court and hoop calibration (manual calibration is accepted for M1).

## 3. Input assumptions

| ID | Requirement |
|---|---|
| IN-1 | Video file in a common container (MP4 / MOV), H.264 or HEVC. |
| IN-2 | **Static camera** for the whole video: no pan, tilt, or zoom. Small vibration is tolerated. |
| IN-3 | **Half court with one hoop.** The rim, free-throw line, and full 3-point arc are visible in frame. |
| IN-4 | Recommended view: elevated sideline or corner position at roughly 45°, so the rim is seen from the side and the shooter's feet are visible. Level sideline views (as in TrackID3x3) are also supported. |
| IN-5 | Resolution ≥ 720p; frame rate ≥ 30 fps (60 fps preferred). 4K input is accepted and may be downscaled internally. |
| IN-6 | Two teams wearing **clearly distinguishable** jersey or bib colors. |
| IN-7 | Video length from a few seconds (short clips) up to 60 minutes. |
| IN-8 | Reasonable lighting (indoor gym or daylight outdoor court); night footage with heavy noise is not a target. |

Videos that violate IN-2, IN-3, or IN-6 are unsupported; the system may reject them or produce degraded results.

## 4. Calibration (manual, once per camera setup)

| ID | Requirement |
|---|---|
| CAL-1 | The user marks the **rim** in a reference frame (e.g. rim center and two points on the rim edge). |
| CAL-2 | The user marks **≥ 4 court landmarks** (e.g. baseline–lane corners, free-throw line ends, 3-point line corners) to compute an image → court homography. |
| CAL-3 | The user selects the **court standard**: NBA, FIBA, FIBA 3x3, NCAA, or high school, because 3-point distances and lane dimensions differ. (FIBA 3x3 uses the FIBA half-court markings with a 6.75 m arc.) |
| CAL-4 | Calibration is saved as a JSON file and can be reused for all videos from the same fixed camera setup (e.g. all clips of one TrackID3x3 subset, if the camera did not move). |
| CAL-5 | A simple calibration tool (click-on-frame UI) is provided; it shows the projected court lines so the user can verify the fit visually. |
| CAL-6 | For datasets that ship court keypoints (TrackID3x3), a converter may produce `calib.json` from those keypoints; manual calibration is then validated against them (see §7.2). |

## 5. Functional requirements — event definitions

Precise definitions matter because evaluation depends on them. Where practical, these follow official
scoring conventions (NBA for 5v5, FIBA for 3x3).

### 5.1 Shot attempt

| ID | Requirement |
|---|---|
| SH-1 | A **shot attempt** is a ball release by a player with the apparent intent to score, i.e. the ball travels toward the rim. |
| SH-2 | Included: jump shots, layups, dunks, hook shots, tip-ins / put-backs, airballs, and **blocked shots**. |
| SH-3 | Excluded: passes (including lob passes that are caught), dribbles, and loose balls. |
| SH-4 | The event **timestamp** is the moment of release (the ball leaves the shooter's hand). |
| SH-5 | Known limitation: shots on which the shooter is fouled and misses count as missed attempts in M1 (fouls are not detected). Official rules would not count these as field-goal attempts. |

### 5.2 Make / miss

| ID | Requirement |
|---|---|
| MK-1 | A shot is **made** when the ball passes downward through the rim. |
| MK-2 | Every other attempt is **missed**, including airballs, blocked shots, and rim-outs. |
| MK-3 | Goaltending and other rule-based rulings are ignored; only the physical outcome counts. |

### 5.3 Shot zone

Shot classification is based on court geometry only and is independent of the rule set.

| ID | Requirement |
|---|---|
| TY-1 | **`ft`** (free throw): an uncontested shot taken from the free-throw line while play is stopped, with other players lined up along the lane or behind the shooter. |
| TY-2 | **`beyond_arc`**: a field goal attempt where the shooter's last ground contact before release is fully **behind** the arc. A foot on the line counts as `inside_arc`. |
| TY-3 | **`inside_arc`**: any other field goal attempt. |
| TY-4 | The shooter's court position (in court coordinates, via the homography) is recorded for every attempt. |

### 5.4 Team attribution

| ID | Requirement |
|---|---|
| TM-1 | Each attempt is attributed to the team of the **shooter**, i.e. the player in possession of the ball at release. |
| TM-2 | Teams are identified by unsupervised clustering of jersey / bib colors; no labels are required. |
| TM-3 | Teams are reported as `A` / `B`. The user may optionally map them to names or colors in a config file. |
| TM-4 | Referees, substitutes, and spectators must not be assigned to a team; they may be ignored. |

### 5.5 Scoring rule sets

| ID | Requirement |
|---|---|
| RS-1 | The rule set is a run parameter: `5v5` (default) or `3x3`. |
| RS-2 | Points per made shot by zone: |

| Zone | `5v5` | `3x3` |
|---|---|---|
| `inside_arc` | 2 | 1 |
| `beyond_arc` | 3 | 2 |
| `ft` | 1 | 1 |

| ID | Requirement |
|---|---|
| RS-3 | Reports use conventional labels per rule set: `2PT / 3PT / FT` for 5v5 and `1PT / 2PT / FT` for 3x3. Internally, and in the event log, zones are always used, to avoid ambiguity. |
| RS-4 | Evaluation of shot classification is performed on zones, so results are comparable across rule sets. |

## 6. Outputs

| ID | Output | Required | Description |
|---|---|---|---|
| OUT-1 | `stats.json` | Yes | Rule set used; per team and in total: attempts, makes, and FG% for each zone and overall; total points. |
| OUT-2 | `events.csv` | Yes | One row per shot attempt: `event_id, timestamp_s, frame, team, zone, made, points, court_x, court_y, confidence`. |
| OUT-3 | `annotated.mp4` | Optional (`--render`) | Input video with overlays: ball / player boxes (colored by team), ball trajectory, rim, and a label at each shot event. Used for debugging and demos. |
| OUT-4 | Run log | Yes | Model versions, parameters, rule set, calibration file used, processing time. |

Interface: a command-line tool, e.g.

```
analyze <video> --calib <calib.json> --rules {5v5,3x3} --out <dir> [--render]
```

## 7. Evaluation & data

### 7.1 Data sources

| Source | Role | Notes |
|---|---|---|
| **TrackID3x3 — Outdoor** ([repo](https://github.com/open-starlab/TrackID3x3), CC BY 4.0) | Primary dev + test | 12 videos, fixed sideline camera (iPhone 13, 4K), outdoor court, 3x3, colored bibs. Ships player boxes (all frames), 10 body keypoints incl. ankles (subset of frames), court keypoints. |
| **TrackID3x3 — Indoor** (same repo and license) | Test only: attempts and make/miss (no team metrics: each player wears a different color). ~20 fps, below IN-5; accepted as a stress test | 42 short clips (~7.5k frames total), fixed sideline camera (720p), university gym, half court. Same annotation types. |
| TrackID3x3 — Drone | Not used | Camera moves; no court keypoints. Violates IN-2. |
| Public ball datasets (e.g. Roboflow Universe, DeepSportradar) | Ball-detector training only | Licenses to be checked before use. |
| Supplementary footage (tripod YouTube videos, self-recorded) | Generalization + FT + 5v5 checks | Needed because TrackID3x3 is 3x3 only, with few free throws and few venues. |

- TrackID3x3 is used under CC BY 4.0: the dataset and paper are credited in the README and in any demo.
- Downloaded third-party footage is used for personal / research purposes only.
- **No video or dataset files are committed to the repository.**

### 7.2 Ground truth

| ID | Requirement |
|---|---|
| EV-1 | Shot events are hand-labeled on TrackID3x3 Outdoor and Indoor videos (the dataset has no ball or shot annotations), plus a small supplementary set. |
| EV-2 | Labels are **event-level** (one row per shot: `timestamp_s, team, zone, made`), not frame-level. |
| EV-3 | A lightweight keyboard-driven labeling tool is provided (play / pause / step, one key per field). |
| EV-4 | After v1 exists, additional videos can be labeled by *reviewing* model-proposed events. All test-set labels must be human-verified. |
| EV-5 | Dev/test splits are made **per video**, never per frame, separately for Outdoor and Indoor. Test videos are never used for training or tuning. |
| EV-6 | Existing TrackID3x3 annotations (player boxes, keypoints, court keypoints, bib identities) are reused as ground truth for component metrics (§7.3). |

### 7.3 Metrics

**End-to-end (primary).** Predicted and ground-truth events are matched one-to-one if their release
timestamps are within **±1.0 s**. Metrics are reported **separately for Outdoor, Indoor, and supplementary footage**, and combined.

| Metric | Definition | Proposed target |
|---|---|---|
| Attempt recall | matched GT attempts / all GT attempts | ≥ 90% |
| Attempt precision | matched predictions / all predictions | ≥ 90% |
| Make/miss accuracy | on matched attempts | ≥ 95% |
| Zone accuracy | `inside_arc` / `beyond_arc` / `ft` on matched attempts | ≥ 90% |
| Team accuracy | on matched attempts | ≥ 90% |
| Points error | \|predicted points − GT points\| per team per video | reported, no target |

Aggregate counts alone are not a sufficient metric, because errors can cancel out; event-level
matching is the primary evaluation.

**Component-level (diagnostic, using TrackID3x3 annotations).**

| Metric | Ground truth | Used in |
|---|---|---|
| Calibration error: mean distance (in court units) between projected and annotated court keypoints | Court keypoints | Phase 1 |
| Player detection precision / recall | Player boxes | Phase 2 |
| Foot localization error on the court (in cm) | Ankle keypoints | Phase 4 |
| Team clustering accuracy per player detection | Bib / player identities | Phase 5 |

### 7.4 Training data

| ID | Requirement |
|---|---|
| TD-1 | Player detection uses a pretrained person detector; TrackID3x3 boxes may be used to fine-tune it if needed (dev videos only). |
| TD-2 | Hoop position comes from calibration; no hoop detector is required in M1. |
| TD-3 | The ball detector is fine-tuned, starting from public labeled basketball datasets (licenses to be checked before use). |
| TD-4 | If public data does not transfer well, label up to a few hundred ball frames from dev videos with a model-assisted annotation tool. |

## 8. Non-functional requirements

| ID | Requirement |
|---|---|
| NF-1 | Runs offline on a MacBook with Apple Silicon (inference via PyTorch MPS or Core ML). |
| NF-2 | Processing time ≤ 3× video duration on the target Mac (e.g. a 10-minute video in ≤ 30 minutes), measured at ≤ 1080p processing resolution. |
| NF-3 | Model training may run locally or on a cloud GPU (e.g. Google Colab); inference must not require the cloud. |
| NF-4 | Results are deterministic for the same input, calibration, rule set, and model version. |
| NF-5 | Large binary files (videos, model weights, datasets) are kept out of git. |

## 9. Open questions

1. Target Mac hardware (chip and RAM), which determines whether training runs locally (affects NF-2, NF-3).
2. Final accuracy targets: the values in 7.3 are proposals to confirm after a first baseline.
3. Tolerance for minor camera movement (e.g. a tripod bumped mid-video): reject, or re-calibrate by segment?
4. How to handle shots at the *other* hoop if a video accidentally includes it: ignore for M1?
5. TrackID3x3 details to verify after download: frame rate; whether the rim is clearly visible from the sideline angle; whether Indoor players wear distinguishable bibs (IN-6); whether each subset uses a single fixed camera position (CAL-4).
6. Indoor clips are short (a few seconds each), so the Indoor subset may contain only a few dozen shots in total. Is that enough as a second-domain test, or do we need more indoor footage?
7. Free throws are rare in 3x3; FT detection may need supplementary 5v5 or shooting-practice footage to be evaluated meaningfully.

## 10. Possible future milestones

- Automatic hoop and court calibration.
- Rebounds (offensive / defensive) and possession tracking.
- Per-player statistics via tracking and jersey-number recognition (TrackID3x3 jersey annotations can help).
- Full-court, drone, and broadcast footage.
- Shot charts and heat maps.
