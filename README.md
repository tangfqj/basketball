# hoopstats

Shot statistics (attempts, makes/misses, shot zone, team) from **fixed-camera** basketball video,
derived purely from the pixels. Demo project — see [`doc/requirements.md`](doc/requirements.md) and
[`doc/plan.md`](doc/plan.md).

## Status

**Milestone 1 complete** (end-to-end pipeline) — results and limitations in
[`doc/milestone1-report.md`](doc/milestone1-report.md). On 5 labelled videos: shots found 100%, precision
97.8%, make/miss 97.1% (97.4% on the untouched IMG_0108), zone 97.7%, team 89.7%.

**Milestone 2 wrapped up** (per-player shots by bib number) — [`doc/milestone2-report.md`](doc/milestone2-report.md).
Shots credited to the right player (team + bib): 83.3% on the same videos (bib reader 99.3% per crop; the main
error is picking the contesting defender as the shooter). New outputs: `players.csv`, shooter box with
team + number in the annotated video.

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
  cli.py              command line: analyze, evaluate, detect, track-ball, calibrate, label, ball-review, shooter-review, bib-crops, ball-dataset, probe, proxy
  pipeline.py         detect -> track ball -> shots + make/miss -> shooter / zone / team / player -> outputs
  render.py           annotated output video (shooter box with team + bib number)
  schema.py           ShotEvent, Zone, Team
  rules.py            5v5 / 3x3 points
  court.py            court standards, zone classification (court coordinates in metres)
  calibration/        calib.json, image <-> court homography, browser calibration tool
  detection/          ball + player detection over a video (cached, batched, optional parallel workers)
  tracking/           ball tracking (one ball per frame, gap filling)
  events/             shot detection, make/miss model, shooter, zone / free throw
  teams/              bib-colour team clustering, team per shot
  players/            bib reader per shot, roster per team, per-player table (Milestone 2)
  outputs.py          events.csv, stats.json, players.csv
  evaluation.py       event matching and metrics
  video.py            ffprobe / ffmpeg helpers
  datasets/           TrackID3x3 loaders
  labeling/           browser tools: shot labeling, ball-box review, shooter review
  training/           training-data export (ball dataset, bib crops) and bib-reader metrics
notebooks/            Colab: ball-detector training; whole pipeline on a cloud GPU; bib-reader training
configs/              dataset splits (dev / test)
labels/               hand-labeled shot events, ball boxes and shooters (committed)
models/               make/miss model (committed; detector and bib-reader weights live in data/models/)
calib/                per-video calibration (committed)
tests/                unit tests
doc/                  requirements, plan, dataset notes
data/                 videos and datasets (git-ignored)
```

## Commands

```bash
# the whole pipeline (detection is cached; first run on a new video: ~45 min on an M2, ~20 min on a Colab A100)
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104              # events.csv, stats.json, players.csv, run_log.json
                                                                     # (players need data/models/bib_v1.pt = bib_final.pt from the bib notebook)
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render     # + annotated.mp4 (whole video)
uv run hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 --render-range 60:120   # annotated clip
uv run hoopstats evaluate --gt labels/IMG_0104.csv --pred out/IMG_0104/events.csv --rules 3x3
uv run python scripts/eval_outputs.py IMG_0104 IMG_0105 IMG_0106 IMG_0107 IMG_0108 --out-dir out   # accuracy table
uv run python scripts/make_colab_bundle.py      # cloud run: see notebooks/run_pipeline_colab.ipynb

uv run hoopstats label data/outdoor/IMG_0104.MOV       # label shots in the browser
uv run hoopstats calibrate data/outdoor/IMG_0104.MOV --init trackid3x3-outdoor   # court + rim
uv run hoopstats ball-review                           # review ball boxes, see doc/ball-labeling-guide.md
uv run hoopstats shooter-review                        # review the shooter of each labeled shot (Milestone 2)
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
