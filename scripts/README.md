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
| `propose_shots.py` | Pre-fill shot labels from the detector (review mode; `--blank-made` for test videos) | shot labelling |
| `eval_shots.py` | Shot attempts and make/miss vs. labels, with an error list | Phase 3 |
| `shot_features.py` | Dump trajectory features of labelled shots | Phase 3 |
| `train_make_model.py` | Leave-one-video-out evaluation and training of the make/miss model | Phase 3 |
| `eval_zones.py` | Shooter position / zone vs. labels | Phase 4 |
| `eval_team_clustering.py` | Team colour clustering vs. TrackID3x3 player boxes | Phase 5, `2026-09-25-team-clustering` |
| `eval_shot_teams.py` | Team per shot vs. labels | Phase 5 |
| `probe_bibs.py` | Contact sheets of dataset player crops with bib numbers (legibility check) | M2 step 0, `2026-09-27-bib-probe` |
| `propose_shooters.py` | Pre-fill the shooter of each labeled shot from dataset boxes; crops for `hoopstats shooter-review` | M2 step 1 |
| `eval_shooters.py` | Shooter accuracy vs. reviewed shooter labels (`--errors` lists misses) | M2 step 2 |
| `shooter_ranker_experiment.py` | Learned candidate ranking for the shooter (negative result) | M2 step 2, `2026-09-27-shooter-accuracy` |
| `train_bib_reader.py` | Train / evaluate the bib reader, one leave-one-video-out fold per run (used by `notebooks/train_bib_reader.ipynb`) | M2 step 3 |
| `eval_players.py` | Player (team + bib) per shot and per-player counts vs. shooter labels | M2 step 4 |
| `player_identity_experiment.py` | Offline comparison of view windows / team-from-bib rules from `cache/<video>/bib_probs.npz` | M2 step 4, `2026-09-27-players-dev` |
