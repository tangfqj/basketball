"""Court geometry and shot-zone classification (requirements CAL-3, TY-2/TY-3).

Court coordinate system (used everywhere in hoopstats):
  * units: metres
  * origin: midpoint of the baseline (end line), on the floor
  * x: along the baseline (positive to the right when facing the basket from the court)
  * y: from the baseline into the court (positive towards half court)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CourtSpec:
    name: str
    width: float          # sideline to sideline
    half_length: float    # baseline to half-court line (3x3: to the top of the court)
    hoop_y: float         # basket centre distance from the baseline
    arc_radius: float     # 3-point arc radius from the basket centre
    corner_x: float       # |x| of the straight corner segments of the 3-point line
    ft_line_y: float      # free-throw line distance from the baseline
    lane_width: float

    @property
    def arc_break_y(self) -> float:
        """y where the straight corner segments meet the arc."""
        return self.hoop_y + math.sqrt(self.arc_radius**2 - self.corner_x**2)


STANDARDS: dict[str, CourtSpec] = {
    "fiba": CourtSpec("fiba", 15.0, 14.0, 1.575, 6.75, 6.60, 5.80, 4.90),
    "fiba_3x3": CourtSpec("fiba_3x3", 15.0, 11.0, 1.575, 6.75, 6.60, 5.80, 4.90),
    "nba": CourtSpec("nba", 15.24, 14.325, 1.60, 7.24, 6.71, 5.79, 4.88),
    # TODO: "ncaa", "high_school"
}


def get_court(name: str) -> CourtSpec:
    try:
        return STANDARDS[name]
    except KeyError:
        raise ValueError(f"unknown court standard {name!r}; expected one of {sorted(STANDARDS)}") from None


def is_beyond_arc(court: CourtSpec, x: float, y: float) -> bool:
    """True if a floor point is strictly outside the 2-point area.

    The 3-point line itself belongs to the 2-point area (TY-2); dimensions are treated as
    measured to the line's outer edge.
    """
    in_corner_strip = abs(x) <= court.corner_x and y <= court.arc_break_y
    # The arc only exists above the point where the straight corner segments end.
    in_arc = y > court.arc_break_y and math.hypot(x, y - court.hoop_y) <= court.arc_radius
    return not (in_corner_strip or in_arc)


def is_on_court(court: CourtSpec, x: float, y: float, margin: float = 0.5) -> bool:
    """Used to filter out people standing off court (TM-4)."""
    return abs(x) <= court.width / 2 + margin and -margin <= y <= court.half_length + margin
