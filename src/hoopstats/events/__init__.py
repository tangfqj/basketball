"""Shot events: attempts, release time, make/miss (Phase 3), zone (Phase 4)."""

from .shots import ShotCandidate, ShotParams, detect_shots

__all__ = ["ShotCandidate", "ShotParams", "classify_zone", "detect_shots"]


def classify_zone(shot, player_detections, calibration, court):
    raise NotImplementedError("Phase 4: zone classification")
