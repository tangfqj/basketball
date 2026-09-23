"""Scoring rule sets (requirements §5.5)."""

from __future__ import annotations

from .schema import Zone

POINTS: dict[str, dict[Zone, int]] = {
    "5v5": {Zone.INSIDE_ARC: 2, Zone.BEYOND_ARC: 3, Zone.FT: 1},
    "3x3": {Zone.INSIDE_ARC: 1, Zone.BEYOND_ARC: 2, Zone.FT: 1},
}

# Conventional report labels per rule set (RS-3).
LABELS: dict[str, dict[Zone, str]] = {
    "5v5": {Zone.INSIDE_ARC: "2PT", Zone.BEYOND_ARC: "3PT", Zone.FT: "FT"},
    "3x3": {Zone.INSIDE_ARC: "1PT", Zone.BEYOND_ARC: "2PT", Zone.FT: "FT"},
}


def points_for(rules: str, zone: Zone, made: bool) -> int:
    if rules not in POINTS:
        raise ValueError(f"unknown rule set {rules!r}; expected one of {sorted(POINTS)}")
    return POINTS[rules][zone] if made else 0
