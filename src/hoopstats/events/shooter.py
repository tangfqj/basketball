"""Shooter, shooting position and zone (Phase 4, requirements TY-1..TY-4, TM-1).

Shooter: the player whose upper body is closest to the ball during the first frames of the flight
(the ball leaves the shooter's hands at the release; a defender's box may overlap, but the ball is at the
shooter's hands/head, not at the defender's). Distances are normalised by box height.

Shooting position: the shooter's floor point (bottom-centre of the box) at the detected release, which is
the start of the shooting motion — feet still planted (last ground contact before a jump shot).
Mapped to court coordinates with the calibration; the zone follows from the court geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..court import CourtSpec, is_beyond_arc, is_on_court


@dataclass
class ShooterInfo:
    track_id: int | None
    frame: int | None                  # frame of the chosen floor point
    foot_px: tuple[float, float] | None
    court_xy: tuple[float, float] | None
    zone: str | None                   # inside_arc / beyond_arc / ft
    score: float | None                # normalised ball-to-hands distance (lower = more certain)


def _persons_at(persons: np.ndarray, f: int) -> np.ndarray:
    return persons[persons[:, 0] == f]


def find_shooter(persons: np.ndarray, track: dict, release: int, person_every: int = 3,
                 look_ahead: int = 9) -> tuple[int | None, float | None]:
    """-> (track id, score). persons: (N, 7) [frame, id, x, y, w, h, conf] in 4K pixels."""
    votes: dict[int, list[float]] = {}
    f0 = release - release % person_every
    for f in range(f0, release + look_ahead + 1, person_every):
        g = f if np.isfinite(track["x"][f]) else None
        if g is None:
            continue
        bx, by = track["x"][g], track["y"][g]
        for _, tid, x, y, w, h, _ in _persons_at(persons, f):
            if tid < 0:
                continue
            hx, hy = x + w / 2, y + 0.15 * h            # hands / head region
            dx = max(0.0, abs(bx - hx) - w / 2)          # inside the box width: no horizontal penalty
            d = np.hypot(dx, max(0.0, by - (y + 0.45 * h)) + max(0.0, (y - 0.4 * h) - by)) / h
            votes.setdefault(int(tid), []).append(d)
    if not votes:
        return None, None
    tid = min(votes, key=lambda t: (np.median(votes[t]), -len(votes[t])))
    return tid, float(np.median(votes[tid]))


def shooting_position(persons: np.ndarray, tid: int, release: int, fps: float,
                      window: int = 3) -> tuple[int, tuple[float, float]] | None:
    """Floor point at the detected release. The detector's release is the start of the shooting motion
    (feet still planted), so the box bottom there is the last ground contact. (Taking the lowest box bottom
    over a longer window picks the point farthest from the basket, e.g. the start of a drive.)"""
    sel = persons[(persons[:, 1] == tid) & (np.abs(persons[:, 0] - release) <= window)]
    if not len(sel):
        sel = persons[(persons[:, 1] == tid) & (np.abs(persons[:, 0] - release) <= 4 * window)]
    if not len(sel):
        return None
    k = int(np.argmin(np.abs(sel[:, 0] - release)))
    f, _, x, y, w, h, _ = sel[k]
    return int(f), (float(x + w / 2), float(y + h))


def at_ft_spot(court_xy: tuple[float, float], court: CourtSpec, tol: float = 0.6) -> bool:
    x, y = court_xy
    return abs(x) <= court.lane_width / 2 * 0.6 and court.ft_line_y - 0.2 <= y <= court.ft_line_y + tol


def classify_zone(court_xy: tuple[float, float], court: CourtSpec, free_throw_context: bool = False) -> str:
    """Free throw = shooter at the free-throw spot *and* free-throw context (standing still, nobody close);
    otherwise the zone follows from the geometry."""
    if free_throw_context and at_ft_spot(court_xy, court):
        return "ft"
    return "beyond_arc" if is_beyond_arc(court, *court_xy) else "inside_arc"


def free_throw_context(persons: np.ndarray, tid: int, release: int, calibration, court: CourtSpec, fps: float,
                       still_s: float = 2.0, max_move_m: float = 0.4, min_gap_m: float = 2.5) -> dict:
    """A free-throw shooter stands still before the shot and nobody is close (players line up along the lane
    or stand behind the arc). On IMG_0104–0108: free throws move < 0.2 m in 2 s with the nearest player
    >= 3.1 m away; in-play shots from the same spot move 3–5 m or have a defender within 1.4–2.6 m."""

    def floor(f, t):
        s = persons[(persons[:, 1] == t) & (np.abs(persons[:, 0] - f) <= 3)]
        if not len(s):
            return None
        s = s[np.argmin(np.abs(s[:, 0] - f))]
        return calibration.image_to_court([(s[2] + s[4] / 2, s[3] + s[5])])[0]

    p0 = floor(release, tid)
    if p0 is None:
        return {"is_ft": False}
    moves = [floor(release - round(k * fps / 2), tid) for k in range(1, int(still_s * 2) + 1)]
    moved = max((float(np.hypot(*(q - p0))) for q in moves if q is not None), default=np.inf)
    gaps = []
    for o in persons[(np.abs(persons[:, 0] - release) <= 1) & (persons[:, 1] != tid)]:
        c = calibration.image_to_court([(o[2] + o[4] / 2, o[3] + o[5])])[0]
        if is_on_court(court, *c, margin=0.3):
            gaps.append(float(np.hypot(*(c - p0))))
    gap = min(gaps, default=np.inf)
    return {"is_ft": moved <= max_move_m and gap >= min_gap_m, "moved_m": moved, "nearest_m": gap}


def shooter_info(persons: np.ndarray, track: dict, release: int, calibration, court: CourtSpec,
                 fps: float) -> ShooterInfo:
    tid, score = find_shooter(persons, track, release)
    if tid is None:
        return ShooterInfo(None, None, None, None, None, None)
    pos = shooting_position(persons, tid, release, fps)
    if pos is None:
        return ShooterInfo(tid, None, None, None, None, score)
    f, foot = pos
    (cx, cy), = calibration.image_to_court([foot])
    ft = at_ft_spot((cx, cy), court) and free_throw_context(persons, tid, release, calibration, court, fps)["is_ft"]
    return ShooterInfo(tid, f, foot, (float(cx), float(cy)), classify_zone((cx, cy), court, ft), score)
