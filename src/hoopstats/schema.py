"""Core data types shared across pipeline stages (requirements §5, §6)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Zone(str, Enum):
    """Shot zone, independent of the scoring rule set (requirements §5.3)."""

    INSIDE_ARC = "inside_arc"
    BEYOND_ARC = "beyond_arc"
    FT = "ft"


class Team(str, Enum):
    A = "A"
    B = "B"
    UNKNOWN = "?"


@dataclass
class ShotEvent:
    """One shot attempt. Mirrors the columns of events.csv (requirements OUT-2)."""

    event_id: int
    timestamp_s: float          # moment of release (SH-4)
    frame: int
    team: Team
    zone: Zone
    made: bool
    points: int = 0             # filled in from the rule set
    court_x: float | None = None  # metres, court coordinates (see court.py)
    court_y: float | None = None
    confidence: float = 1.0
    player: str = ""            # bib number of the shooter, "?" if unreadable (Milestone 2, PL-1/PL-5)
    shooter_track: int | None = None   # internal player track id

    def to_row(self) -> dict:
        row = asdict(self)
        row["team"] = self.team.value
        row["zone"] = self.zone.value
        row["made"] = int(self.made)
        return row


EVENT_COLUMNS = [
    "event_id", "timestamp_s", "frame", "team", "zone", "made",
    "points", "court_x", "court_y", "confidence", "player", "shooter_track",
]
