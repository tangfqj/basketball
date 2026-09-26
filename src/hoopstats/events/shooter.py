"""Shooter, shooting position and zone (Phase 4, requirements TY-1..TY-4, TM-1).

Shooter: the player whose upper body is closest to the ball during the first frames of the flight.
(Ranking by ball possession before the release was tried and made results worse; see find_shooter.)

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


def _near_release(persons, track, release, person_every, look_ahead):
    """Old criterion: normalised distance from the ball to each player's upper body just after release."""
    votes: dict[int, list[float]] = {}
    f0 = release - release % person_every
    for f in range(f0, release + look_ahead + 1, person_every):
        if not np.isfinite(track["x"][f]):
            continue
        bx, by = track["x"][f], track["y"][f]
        for _, tid, x, y, w, h, _ in _persons_at(persons, f):
            if tid < 0:
                continue
            hx = x + w / 2                               # hands / head region: upper part of the box
            dx = max(0.0, abs(bx - hx) - w / 2)          # inside the box width: no horizontal penalty
            d = np.hypot(dx, max(0.0, by - (y + 0.45 * h)) + max(0.0, (y - 0.4 * h) - by)) / h
            votes.setdefault(int(tid), []).append(d)
    return {t: (float(np.median(v)), len(v)) for t, v in votes.items()}


def _possession(persons, track, balls, release, person_every, before):
    """Frames in the `before` window in which a ball position (track, or a raw detection) lies inside a
    player's box — the shooter has the ball before the shot; a contesting defender only near the release."""
    counts: dict[int, int] = {}
    f0 = release - before
    f0 -= f0 % person_every
    for f in range(f0, release + 1, person_every):
        pts = []
        if np.isfinite(track["x"][f]):
            pts.append((track["x"][f], track["y"][f]))
        if balls is not None:
            sel = balls[(balls[:, 0] == f) & (balls[:, 5] >= 0.15)]
            pts += [(b[1] + b[3] / 2, b[2] + b[4] / 2) for b in sel]
        if not pts:
            continue
        for _, tid, x, y, w, h, _ in _persons_at(persons, f):
            if tid < 0:
                continue
            mx = 0.1 * w
            if any(x - mx <= px <= x + w + mx and y - 0.1 * h <= py <= y + 0.8 * h for px, py in pts):
                counts[int(tid)] = counts.get(int(tid), 0) + 1
    return counts


def find_shooter(persons: np.ndarray, track: dict, release: int, person_every: int = 3,
                 look_ahead: int = 9, balls: np.ndarray | None = None, before: int = 12,
                 use_possession: bool = False) -> tuple[int | None, float | None]:
    """-> (track id, score): the player whose upper body is closest to the ball just after release.
    `use_possession` (experimental, off): rank first by ball possession in the `before` frames. It made both
    zone (97.7% -> 93-96%) and team (IMG_0107: 80% -> 74%) worse on IMG_0104-0108 — in crowded shots the
    contesting defender's box also contains the ball, and longer windows reach back to the passer.
    persons: (N, 7) [frame, id, x, y, w, h, conf]; balls: raw ball detections (N, 7), optional."""
    near = _near_release(persons, track, release, person_every, look_ahead)
    poss = _possession(persons, track, balls, release, person_every, before) if use_possession else {}
    cands = list(near) + [t for t in poss if t not in near]      # deterministic order for exact ties
    if not cands:
        return None, None
    # ties (several boxes at distance 0) go to the player seen in more frames
    tid = min(cands, key=lambda t: (-poss.get(t, 0), near.get(t, (np.inf, 0))[0], -near.get(t, (np.inf, 0))[1]))
    return tid, near[tid][0] if tid in near else None


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
                 fps: float, balls: np.ndarray | None = None) -> ShooterInfo:
    tid, score = find_shooter(persons, track, release, balls=balls)
    if tid is None:
        return ShooterInfo(None, None, None, None, None, None)
    pos = shooting_position(persons, tid, release, fps)
    if pos is None:
        return ShooterInfo(tid, None, None, None, None, score)
    f, foot = pos
    (cx, cy), = calibration.image_to_court([foot])
    ft = at_ft_spot((cx, cy), court) and free_throw_context(persons, tid, release, calibration, court, fps)["is_ft"]
    return ShooterInfo(tid, f, foot, (float(cx), float(cy)), classify_zone((cx, cy), court, ft), score)
