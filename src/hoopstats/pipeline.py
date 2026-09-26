"""End-to-end pipeline: detect -> track -> shots -> shooter / zone / team -> outputs (Phase 6).

  hoopstats analyze data/outdoor/IMG_0104.MOV --out out/IMG_0104 [--render]

Each stage caches its result under <cache_dir>/<video stem>/ (detections, ball_track.npz, keyframes),
so re-running only recomputes what is missing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .calibration import Calibration
from .court import get_court
from .detection.runner import load_detections, run_detection
from .events import detect_shots
from .events.make_model import MakeModel
from .events.shooter import shooter_info
from .outputs import apply_rules, write_events_csv, write_stats_json
from .schema import ShotEvent, Team, Zone
from .teams.assign import fit_video_team_model, shot_team
from .tracking import track_ball
from .video import probe

RELEASE_SHIFT_S = 0.40   # the detected rise starts ~0.4 s before the ball leaves the hands (IMG_0104 labels)


@dataclass
class RunConfig:
    video: str
    calib: str
    out_dir: str
    rules: str = "3x3"
    cache_dir: str = "cache"
    ball_model: str = "data/models/ball_v1.pt"
    person_model: str = "data/models/yolo11s.pt"
    make_model: str = "models/make_model.json"
    render: bool = False
    render_range: tuple[float, float] | None = None     # seconds
    batch: int = 1                                        # frames per model call (8-16 on a large GPU)
    device: str | None = None


def _detections_complete(cache: Path, n_frames: int, chunk: int = 900) -> bool:
    return all((cache / "detections" / f"chunk_{c:06d}.npz").exists() for c in range(0, n_frames, chunk))


def analyze(cfg: RunConfig) -> list[ShotEvent]:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg.cache_dir) / Path(cfg.video).stem
    info = probe(cfg.video)
    fps = info.fps
    cal = Calibration.load(cfg.calib)
    court = get_court(cal.court_standard)
    rr = abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2 if len(cal.rim_edge) >= 2 else 33.0
    log = {"version": __version__, "video": cfg.video, "calib": cfg.calib, "rules": cfg.rules,
           "court_standard": cal.court_standard, "ball_model": cfg.ball_model, "make_model": cfg.make_model}

    stage = {}
    ts = time.time()

    def lap(name):
        nonlocal ts
        stage[name] = round(time.time() - ts, 1)
        ts = time.time()

    # 1. detection (cached, resumable)
    if not _detections_complete(cache, info.n_frames):
        print("[1/5] detecting ball and players ...")
        run_detection(cfg.video, cfg.calib, cfg.ball_model, cfg.person_model, cfg.cache_dir,
                      device=cfg.device, batch=cfg.batch)
    else:
        print("[1/5] detections found in cache")
    balls, persons = load_detections(cache)
    lap("detect")

    # 2. ball track (cached)
    track_p = cache / "ball_track.npz"
    if track_p.exists():
        track = dict(np.load(track_p))
    else:
        print("[2/5] tracking the ball ...")
        track = track_ball(balls, info.n_frames)
        np.savez_compressed(track_p, **track)
    lap("track")

    # 3. shots + make/miss
    print("[3/5] shots and make/miss ...")
    mm = MakeModel.load(cfg.make_model) if cfg.make_model and Path(cfg.make_model).exists() else None
    shots = detect_shots(track, cal.rim_center, rr, fps, make_model=mm)
    lap("shots")

    # 4. shooter, zone, team
    print(f"[4/5] shooter, zone and team for {len(shots)} shots ...")
    team_model, bg = fit_video_team_model(cfg.video, persons, cal, court, cache)
    cap = cv2.VideoCapture(cfg.video)
    events = []
    for i, s in enumerate(shots):
        si = shooter_info(persons, track, s.release_frame, cal, court, fps, balls=balls)
        team = Team.UNKNOWN
        if si.track_id is not None:
            t, _ = shot_team(cap, persons, si.track_id, s.release_frame, team_model, bg)
            team = Team(t)
        zone = Zone(si.zone) if si.zone else Zone.INSIDE_ARC
        release = s.release_frame + round(RELEASE_SHIFT_S * fps)
        events.append(ShotEvent(
            event_id=i, timestamp_s=round(release / fps, 3), frame=release, team=team, zone=zone, made=s.made,
            court_x=None if si.court_xy is None else round(si.court_xy[0], 2),
            court_y=None if si.court_xy is None else round(si.court_xy[1], 2),
            confidence=round(s.confidence, 3)))

    lap("shooter_zone_team")

    # 5. outputs
    print("[5/5] writing outputs ...")
    apply_rules(events, cfg.rules)
    write_events_csv(events, out / "events.csv")
    write_stats_json(events, cfg.rules, out / "stats.json")
    if cfg.render:
        from .render import render_video

        render_video(cfg.video, track, persons, events, shots, cal, out / "annotated.mp4", fps, cfg.rules,
                     cfg.render_range)
    lap("outputs_render")
    timing_p = cache / "detections" / "timing.json"
    log.update(n_shots=len(events), video_s=round(info.duration_s, 1), processing_s=round(time.time() - t0, 1),
               stage_s=stage, detection=json.loads(timing_p.read_text()) if timing_p.exists() else None)
    (out / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"done: {len(events)} shots -> {out}")
    return events
