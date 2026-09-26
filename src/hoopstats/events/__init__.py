"""Shot events (Phases 3-4): attempts, release time, make/miss; shooter and zone live in events/shooter.py."""

from .shots import ShotCandidate, ShotParams, detect_shots

__all__ = ["ShotCandidate", "ShotParams", "detect_shots"]
