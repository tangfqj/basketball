"""End-to-end pipeline: calibrate -> detect -> track -> events -> outputs.

Each stage will write its intermediate result to <cache_dir>/<video stem>/<stage>.* so later stages
can be iterated without re-running detection (see doc/plan.md, "Cached stages").
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .calibration import Calibration
from .court import get_court
from .detection.ball import detect_ball
from .detection.players import detect_players
from .events import detect_shots
from .outputs import apply_rules, write_events_csv, write_stats_json
from .teams import assign_teams
from .tracking import track_ball
from .video import probe


@dataclass
class RunConfig:
    video: str
    calib: str
    out_dir: str
    rules: str = "3x3"
    cache_dir: str = "cache"
    render: bool = False


def analyze(cfg: RunConfig) -> None:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg.cache_dir) / Path(cfg.video).stem
    cache.mkdir(parents=True, exist_ok=True)

    info = probe(cfg.video)
    calib = Calibration.load(cfg.calib)
    court = get_court(calib.court_standard)

    players = detect_players(cfg.video, str(cache))               # Phase 2.1
    balls = detect_ball(cfg.video, str(cache))                    # Phase 2.2
    tracks = track_ball(balls, info.fps)                          # Phase 2.3
    teams = assign_teams(cfg.video, players, calib, court)        # Phase 5
    events = detect_shots(tracks, calib, info.fps)                # Phases 3-4
    del teams  # TODO: attach teams to events

    apply_rules(events, cfg.rules)
    write_events_csv(events, out / "events.csv")
    write_stats_json(events, cfg.rules, out / "stats.json")
    if cfg.render:
        raise NotImplementedError("Phase 6.1: annotated video")

    (out / "run_log.json").write_text(json.dumps({
        "version": __version__, "video": cfg.video, "calib": cfg.calib, "rules": cfg.rules,
        "court_standard": calib.court_standard, "processing_s": round(time.time() - t0, 1),
    }, indent=2))
