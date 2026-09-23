"""Object detection (Phase 2): players and ball."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    frame: int
    x: float        # top-left, pixels
    y: float
    w: float
    h: float
    score: float
    cls: str        # "person" | "ball"
    track_id: int | None = None

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: approximate floor contact point."""
        return (self.x + self.w / 2, self.y + self.h)
