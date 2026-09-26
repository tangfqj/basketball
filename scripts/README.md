# Scripts

One-off and evaluation scripts. Library code lives in `src/hoopstats/`; run with `uv run python scripts/<name>.py`.

| Script | Purpose | Used in |
|---|---|---|
| `make_colab_bundle.py` | Pack code, calibrations, labels and models for the Colab notebooks | cloud runs |
| `eval_outputs.py` | Accuracy + speed table for `hoopstats analyze` outputs vs. labels | final evaluation |
| `derive_calibration.py` | Transfer a calibration to another video from the same (slightly moved) camera | calibration |
| `check_outdoor_calibration.py` | Overlay the court model from the TrackID3x3 keypoints (first visual check) | Phase 1 |
| `probe_ball_detection.py` | Pretrained-YOLO feasibility probe on labelled shots | Phase 2, `2026-09-25-ball-probe` |
| `find_ball_flight.py` | Cheap "ball in flight near the rim" signal to find shot moments | ball labelling |
| `select_ball_frames.py` | Pick and extract frames for ball-box labelling | ball labelling |
| `propose_ball_boxes.py` | Pre-annotate ball boxes with the pretrained detector | ball labelling |
| `eval_ball_tracking.py` | Tracking recall / shot coverage vs. reviewed ball boxes | Phase 2 |
| `render_ball_track.py` | Debug video of the ball track for a frame range | Phase 2 |
| `propose_shots.py` | Pre-fill shot labels from the detector (review mode; `--blank-made` for test videos) | shot labelling |
| `eval_shots.py` | Shot attempts and make/miss vs. labels, with an error list | Phase 3 |
| `shot_features.py` | Dump trajectory features of labelled shots | Phase 3 |
| `train_make_model.py` | Leave-one-video-out evaluation and training of the make/miss model | Phase 3 |
| `eval_zones.py` | Shooter position / zone vs. labels | Phase 4 |
| `eval_team_clustering.py` | Team colour clustering vs. TrackID3x3 player boxes | Phase 5, `2026-09-25-team-clustering` |
| `eval_shot_teams.py` | Team per shot vs. labels | Phase 5 |
