# hoopstats

Shot statistics (attempts, makes/misses, shot zone, team) from **fixed-camera** basketball video,
derived purely from the pixels. Demo project — see [`doc/requirements.md`](doc/requirements.md) and
[`doc/plan.md`](doc/plan.md).

## Status

Milestone 1, Phase 6 (end-to-end pipeline working; speed and final test pending). Implemented: rule sets, court geometry and zone classification,
calibration file + homography, event I/O and stats, evaluation, TrackID3x3 loaders, video probe/proxy,
shot labeling tool.
Detection, tracking, shot events and team clustering are stubs (`NotImplementedError`).

## Setup (macOS, Apple Silicon)

```bash
brew install uv ffmpeg          # once
cd ~/Code/basketball
uv sync                         # creates .venv with Python deps
uv run pytest                   # run the tests
uv run hoopstats --help
```

The ML dependencies (PyTorch, detector) are added in Phase 2: `uv sync --extra ml`.

## Layout

```
src/hoopstats/
  cli.py              command-line entry point (probe, proxy, calibrate, analyze, evaluate)
  pipeline.py         calibrate -> detect -> track -> events -> outputs
  schema.py           ShotEvent, Zone, Team
  rules.py            5v5 / 3x3 points
  court.py            court standards, zone classification (court coordinates in metres)
  calibration/        calib.json, image <-> court homography, calibration tool (TODO)
  detection/          player and ball detectors (TODO)
  tracking/           ball tracking (TODO)
  events/             shot detection, make/miss, zones (TODO)
  teams/              team colour clustering (TODO)
  outputs.py          events.csv, stats.json
  evaluation.py       event matching and metrics
  video.py            ffprobe / ffmpeg helpers
  datasets/           TrackID3x3 loaders
  labeling/           browser tools: shot labeling, ball-box review
  training/           training-data export (ball dataset)
notebooks/            Colab notebook for training the ball detector
configs/              default run configuration, dataset splits
labels/               hand-labeled shot events and ball boxes (committed)
calib/                per-video calibration (committed)
tests/                unit tests
doc/                  requirements, plan, dataset notes
data/                 videos and datasets (git-ignored)
```

## Commands

```bash
# the whole pipeline (detection is cached; first run on a new video takes ~45 min on an M2)
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104              # events.csv, stats.json, run_log.json
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render     # + annotated.mp4 (whole video)
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render-range 60:120   # annotated clip
uv run hoopstats evaluate --gt labels/IMG_0104.csv --pred out/IMG_0104/events.csv --rules 3x3

uv run hoopstats label data/outdoor/IMG_0104.MOV       # label shots in the browser, see doc/labeling-guide.md
uv run hoopstats calibrate data/outdoor/IMG_0104.MOV --init trackid3x3-outdoor   # court + rim
uv run hoopstats ball-review                           # review ball boxes, see doc/ball-labeling-guide.md
uv run hoopstats ball-dataset                          # -> data/ball_dataset.zip for Colab training
uv run hoopstats probe data/outdoor/*.MOV
uv run hoopstats proxy data/outdoor/IMG_0104.MOV cache/IMG_0104/proxy_1080.mp4
uv run hoopstats evaluate --gt labels/IMG_0104.csv --pred out/IMG_0104/events.csv --rules 3x3
```

## Data

Videos and datasets live in `data/` and are never committed.

- TrackID3x3 (CC BY 4.0): videos from the dataset's Google Drive into `data/outdoor/` and `data/indoor/`;
  annotations: `git clone https://github.com/open-starlab/TrackID3x3 data/trackid3x3_repo`.

## Attribution

Uses the TrackID3x3 dataset: K. Yamada et al., *TrackID3x3: A Dataset for 3x3 Basketball Player Tracking
and Identification*, MMSports '25 (arXiv:2503.18282). Licensed CC BY 4.0.
