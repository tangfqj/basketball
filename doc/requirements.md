# Basketball Shot Analytics — Requirements (Milestone 1)

| | |
|---|---|
| **Status** | Draft v0.1 |
| **Owner** | Kevin |
| **Last updated** | 2026-09-23 |

## 1. Purpose

Build a demo system that takes a fixed-camera basketball video and outputs shot statistics
(attempts, makes, misses, split by shot type and team) **derived purely from visual understanding of
the video**. The project is a demonstration of video understanding, so reading a scoreboard, captions,
or any other overlaid text is explicitly out of scope as a source of truth.

## 2. Scope

### 2.1 In scope (Milestone 1)

- Detect every **shot attempt** and classify it as **made** or **missed**.
- Classify each attempt as **2PT**, **3PT**, or **FT** (free throw).
- Attribute each attempt to one of **two teams**.
- Produce aggregated statistics and a per-shot event log.
- Offline (batch) processing on a local machine.

### 2.2 Out of scope (Milestone 1)

- Broadcast / TV footage (moving camera, cuts, replays, graphics).
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
| IN-4 | Recommended view: elevated sideline or corner position at roughly 45°, so the rim is seen from the side and the shooter's feet are visible. |
| IN-5 | Resolution ≥ 720p; frame rate ≥ 30 fps (60 fps preferred). |
| IN-6 | Two teams wearing **clearly distinguishable** jersey colors. |
| IN-7 | Video length up to 60 minutes. |
| IN-8 | Reasonable lighting (indoor gym or daylight outdoor court); night footage with heavy noise is not a target. |

Videos that violate IN-2, IN-3, or IN-6 are unsupported; the system may reject them or produce degraded results.

## 4. Calibration (manual, once per video)

| ID | Requirement |
|---|---|
| CAL-1 | The user marks the **rim** in a reference frame (e.g. rim center and two points on the rim edge). |
| CAL-2 | The user marks **≥ 4 court landmarks** (e.g. baseline–lane corners, free-throw line ends, 3-point line corners) to compute an image → court homography. |
| CAL-3 | The user selects the **court standard** (NBA / FIBA / NCAA / high school), because 3-point distances and lane dimensions differ. |
| CAL-4 | Calibration is saved as a JSON file alongside the video and can be reused for other videos from the same camera setup. |
| CAL-5 | A simple calibration tool (click-on-frame UI) is provided; it shows the projected court lines so the user can verify the fit visually. |

## 5. Functional requirements — event definitions

Precise definitions matter because evaluation depends on them. Where practical, these follow NBA scoring conventions.

### 5.1 Shot attempt

| ID | Requirement |
|---|---|
| SH-1 | A **shot attempt** is a ball release by a player with the apparent intent to score, i.e. the ball travels toward the rim. |
| SH-2 | Included: jump shots, layups, dunks, hook shots, tip-ins / put-backs, airballs, and **blocked shots**. |
| SH-3 | Excluded: passes (including lob passes that are caught), dribbles, and loose balls. |
| SH-4 | The event **timestamp** is the moment of release (the ball leaves the shooter's hand). |
| SH-5 | Known limitation: shots on which the shooter is fouled and misses count as missed attempts in M1 (fouls are not detected). NBA rules would not count these as field-goal attempts. |

### 5.2 Make / miss

| ID | Requirement |
|---|---|
| MK-1 | A shot is **made** when the ball passes downward through the rim. |
| MK-2 | Every other attempt is **missed**, including airballs, blocked shots, and rim-outs. |
| MK-3 | Goaltending and other rule-based rulings are ignored; only the physical outcome counts. |

### 5.3 Shot type

| ID | Requirement |
|---|---|
| TY-1 | **FT**: an uncontested shot taken from the free-throw line while play is stopped, with other players lined up along the lane or behind the shooter. |
| TY-2 | **3PT**: a field goal attempt where the shooter's last ground contact before release is fully **behind** the 3-point line. A foot on the line counts as 2PT. |
| TY-3 | **2PT**: any other field goal attempt. |
| TY-4 | The shooter's court position (in court coordinates, via the homography) is recorded for every attempt. |

### 5.4 Team attribution

| ID | Requirement |
|---|---|
| TM-1 | Each attempt is attributed to the team of the **shooter**, i.e. the player in possession of the ball at release. |
| TM-2 | Teams are identified by unsupervised clustering of jersey colors; no labels are required. |
| TM-3 | Teams are reported as `A` / `B`. The user may optionally map them to names or colors in a config file. |
| TM-4 | Referees and spectators must not be assigned to a team; they may be ignored. |

## 6. Outputs

| ID | Output | Required | Description |
|---|---|---|---|
| OUT-1 | `stats.json` | Yes | Aggregated counts per team and in total: attempts, makes, and FG% for 2PT, 3PT, FT, and overall; total points. |
| OUT-2 | `events.csv` | Yes | One row per shot attempt: `event_id, timestamp_s, frame, team, shot_type, made, court_x, court_y, confidence`. |
| OUT-3 | `annotated.mp4` | Optional (`--render`) | Input video with overlays: ball / player boxes (colored by team), ball trajectory, rim, and a label at each shot event. Used for debugging and demos. |
| OUT-4 | Run log | Yes | Model versions, parameters, calibration file used, processing time. |

Interface: a command-line tool, e.g.

```
analyze <video> --calib <calib.json> --out <dir> [--render]
```

## 7. Evaluation & data

### 7.1 Ground truth

| ID | Requirement |
|---|---|
| EV-1 | Build a hand-labeled test set of **3–5 clips** (≈ 15–30 minutes each) from different courts or camera setups. |
| EV-2 | Labels are **event-level** (one row per shot, same schema as OUT-2 minus confidence/coordinates), not frame-level. |
| EV-3 | A lightweight keyboard-driven labeling tool is provided (play / pause / step, one key per field). |
| EV-4 | After v1 exists, additional clips can be labeled by *reviewing* model-proposed events. All test-set labels must be human-verified. |
| EV-5 | Test clips are never used for training or threshold tuning; keep a separate dev set for tuning. |

### 7.2 Metrics

Predicted and ground-truth events are matched one-to-one if their release timestamps are within **±1.0 s**.

| Metric | Definition | Proposed target |
|---|---|---|
| Attempt recall | matched GT attempts / all GT attempts | ≥ 90% |
| Attempt precision | matched predictions / all predictions | ≥ 90% |
| Make/miss accuracy | on matched attempts | ≥ 95% |
| Shot-type accuracy | 2PT / 3PT / FT on matched attempts | ≥ 90% |
| Team accuracy | on matched attempts | ≥ 90% |
| Points error | \|predicted points − GT points\| per team per clip | reported, no target |

Aggregate counts alone are not a sufficient metric, because errors can cancel out; event-level
matching is the primary evaluation.

### 7.3 Training data

| ID | Requirement |
|---|---|
| TD-1 | Player detection uses a pretrained person detector; no custom labels are expected. |
| TD-2 | Hoop position comes from calibration; no hoop detector is required in M1. |
| TD-3 | The ball detector is fine-tuned, starting from public labeled basketball datasets (licenses to be checked before use). |
| TD-4 | If public data does not transfer well, label up to a few hundred frames from our own footage with a model-assisted annotation tool. |

### 7.4 Footage sourcing

- Collect footage from public sources (e.g. tripod-filmed pickup / 3x3 / half-court games, shooting workouts) and optionally self-recorded sessions.
- Verify each candidate video satisfies Section 3 (in particular, that the camera is truly static).
- Downloaded footage is used for personal / research purposes only and is **not** committed to the repository or redistributed.

## 8. Non-functional requirements

| ID | Requirement |
|---|---|
| NF-1 | Runs offline on a MacBook with Apple Silicon (inference via PyTorch MPS or Core ML). |
| NF-2 | Processing time ≤ 3× video duration on the target Mac (e.g. a 10-minute video in ≤ 30 minutes). |
| NF-3 | Model training may run locally or on a rented cloud GPU; inference must not require the cloud. |
| NF-4 | Results are deterministic for the same input, calibration, and model version. |
| NF-5 | Large binary files (videos, model weights, datasets) are kept out of git. |

## 9. Open questions

1. Target Mac hardware (chip and RAM), which determines whether training runs locally (affects NF-2, NF-3).
2. Final accuracy targets: the values in 7.2 are proposals to confirm after a first baseline.
3. Tolerance for minor camera movement (e.g. a tripod bumped mid-video): reject, or re-calibrate by segment?
4. How to handle shots at the *other* hoop if a video accidentally includes it: ignore for M1?

## 10. Possible future milestones

- Automatic hoop and court calibration.
- Rebounds (offensive / defensive) and possession tracking.
- Per-player statistics via tracking and jersey-number recognition.
- Full-court and broadcast footage.
- Shot charts and heat maps.
