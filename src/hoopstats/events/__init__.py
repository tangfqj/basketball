"""Shot events: attempts, release time, make/miss, zone (Phases 3-4). TODO."""


def detect_shots(ball_tracks, calibration, fps: float):
    raise NotImplementedError("Phase 3.1: shot detection")


def classify_make(shot, ball_tracks, calibration) -> bool:
    raise NotImplementedError("Phase 3.2: make/miss")


def classify_zone(shot, player_detections, calibration, court):
    raise NotImplementedError("Phase 4: zone classification")
