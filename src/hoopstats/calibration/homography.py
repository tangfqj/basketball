"""Calibration file (calib.json) and image -> court homography."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    court_standard: str
    image_size: tuple[int, int]                 # (width, height) of the frame the points refer to
    image_points: list[tuple[float, float]]     # pixel coordinates of landmarks
    court_points: list[tuple[float, float]]     # the same landmarks in court coordinates (metres)
    rim_center: tuple[float, float] | None = None   # pixels (CAL-1)
    rim_edge: list[tuple[float, float]] = field(default_factory=list)  # pixels: left, right, front, back
    note: str = ""
    landmark_names: list[str] = field(default_factory=list)  # optional, same order as image_points

    def __post_init__(self) -> None:
        if len(self.image_points) != len(self.court_points) or len(self.image_points) < 4:
            raise ValueError("need >= 4 matching image/court landmark pairs (CAL-2)")
        src = np.asarray(self.image_points, dtype=np.float64)
        dst = np.asarray(self.court_points, dtype=np.float64)
        self._H, _ = cv2.findHomography(src, dst)
        self._H_inv = np.linalg.inv(self._H)

    # --- mapping -------------------------------------------------------------------------
    def image_to_court(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) pixel coordinates -> (N, 2) court coordinates in metres."""
        return _apply(self._H, pts)

    def court_to_image(self, pts: np.ndarray) -> np.ndarray:
        return _apply(self._H_inv, pts)

    def reprojection_error(self) -> float:
        """Mean distance (metres) between mapped image landmarks and their court positions."""
        mapped = self.image_to_court(np.asarray(self.image_points))
        return float(np.linalg.norm(mapped - np.asarray(self.court_points), axis=1).mean())

    # --- io ------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        d = {k: getattr(self, k) for k in
             ("court_standard", "image_size", "image_points", "court_points", "rim_center", "rim_edge", "note",
                  "landmark_names")}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        d["image_size"] = tuple(d["image_size"])
        d["image_points"] = [tuple(p) for p in d["image_points"]]
        d["court_points"] = [tuple(p) for p in d["court_points"]]
        if d.get("rim_center") is not None:
            d["rim_center"] = tuple(d["rim_center"])
        d["rim_edge"] = [tuple(p) for p in d.get("rim_edge", [])]
        return cls(**d)

    def scaled(self, width: int, height: int) -> Calibration:
        """Same calibration for a resized copy of the video (e.g. the 1080p working copy)."""
        sx, sy = width / self.image_size[0], height / self.image_size[1]

        def sc(p):
            return (p[0] * sx, p[1] * sy)

        return Calibration(self.court_standard, (width, height), [sc(p) for p in self.image_points],
                           list(self.court_points), sc(self.rim_center) if self.rim_center else None,
                           [sc(p) for p in self.rim_edge], self.note, list(self.landmark_names))


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
