"""Shot attempts and make/miss from the ball track (Phase 3, requirements §5.1–5.2).

Image geometry (4K pixels): rim centre (rx, ry) and rim radius rr from the calibration; dy > 0 is below
the rim level in the image.

Shot attempt = an *upward flight* of the ball that
  * starts clearly below the rim (from a player's hands — rim bounces start at rim level and do not count),
  * rises by a minimum height, and
  * peaks above the rim level, and
  * either peaks near the basket or comes down close to the rim (far corner shots peak far to the side).
Release = the first frame of that upward flight (requirements SH-4).

Make/miss (v1, rule-based; features are kept for a learned classifier later):
  * the ball crosses the rim level *downwards* close to the rim centre (|dx| small),
  * keeps falling through the net region below the rim, staying roughly under the rim,
  * and does not bounce back above the rim right after the crossing.
Known risk (head-on camera): a ball dropping just *in front of* the rim looks similar; its larger apparent
size at the crossing is recorded as a feature (`size_ratio`) for later use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShotParams:
    min_rise_px: float = 150.0        # vertical rise of the ball during the upward flight
    start_below_rr: float = 2.5       # flight must start at least this many rim radii below the rim
    apex_above_px: float = 10.0       # apex at least this far above the rim level ...
    apex_max_dx_rr: float = 9.0       # ... and within this horizontal distance (rim radii)
    near_rim_dx_rr: float = 4.0       # or: the descent comes this close to the rim
    near_rim_dy_rr: float = 3.0
    min_up_frames: int = 4
    max_gap: int = 3                  # frames without a position tolerated inside a flight
    descent_window_s: float = 1.5     # look for the rim crossing within this time after the apex
    # make rule
    make_dx_rr: float = 0.85          # |dx| at the rim-level crossing
    net_depth_rr: float = 2.5         # ball must reach this far below the rim ...
    net_dx_rr: float = 1.6            # ... while staying under the rim
    net_window_s: float = 0.5
    bounce_rr: float = 0.5            # rising back above rim level by this much after crossing = rim-out


@dataclass
class ShotCandidate:
    release_frame: int
    apex_frame: int
    cross_frame: int | None
    made: bool
    confidence: float
    features: dict = field(default_factory=dict)


def _segments_rising(y: np.ndarray, valid: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Maximal frame ranges [s, e] in which the ball keeps moving up in the image (y decreasing)."""
    idx = np.nonzero(valid)[0]
    segs, s, prev = [], None, None
    for f in idx:
        if prev is not None and f - prev <= max_gap + 1 and y[f] < y[prev] - 0.5:
            if s is None:
                s = prev
        else:
            if s is not None:
                segs.append((s, prev))
            s = None
        prev = f
    if s is not None:
        segs.append((s, prev))
    return segs


def detect_shots(track: dict, rim_center: tuple[float, float], rim_radius: float, fps: float,
                 params: ShotParams | None = None) -> list[ShotCandidate]:
    p = params or ShotParams()
    x, y, size, state = track["x"], track["y"], track["size"], track["state"]
    valid = state > 0
    rx, ry = rim_center
    rr = rim_radius
    expected_size = 2 * rr * 0.53          # ball (24 cm) vs rim (45 cm) at rim depth
    shots: list[ShotCandidate] = []
    for s, e in _segments_rising(y, valid, p.max_gap):
        if e - s + 1 < p.min_up_frames or y[s] - y[e] < p.min_rise_px:
            continue
        if y[s] - ry < p.start_below_rr * rr:                # starts at rim level: rim bounce / tip
            continue
        apex = e
        # the descent after the apex
        end = min(len(y), apex + int(p.descent_window_s * fps))
        after = [f for f in range(apex, end) if valid[f]]
        near = any(abs(x[f] - rx) <= p.near_rim_dx_rr * rr and abs(y[f] - ry) <= p.near_rim_dy_rr * rr for f in after)
        above = ry - y[apex] >= p.apex_above_px                  # a shot must go above the rim level
        if not (above and (near or abs(x[apex] - rx) <= p.apex_max_dx_rr * rr)):
            continue
        # a new shot must not start while the previous one is still on its way to the rim
        if shots and s <= (shots[-1].cross_frame or shots[-1].apex_frame):
            continue
        # rim-level crossing on the way down
        cross = None
        for f0, f1 in zip(after, after[1:]):
            if y[f0] < ry <= y[f1]:
                cross = f1
                t = (ry - y[f0]) / max(1e-6, y[f1] - y[f0])
                dx_cross = float(x[f0] + t * (x[f1] - x[f0]) - rx)
                break
        feats = {"rise_px": float(y[s] - y[e]), "apex_dy": float(y[apex] - ry), "apex_dx": float(x[apex] - rx),
                 "up_frames": int(e - s + 1)}
        made, conf = False, 0.5
        if cross is None:
            feats["crossing"] = "none"
            conf = 0.7                                           # never came down through rim level near the basket
        else:
            win = [f for f in range(cross, min(len(y), cross + int(p.net_window_s * fps))) if valid[f]]
            through = any(y[f] - ry >= p.net_depth_rr * rr for f in win if abs(x[f] - rx) <= p.net_dx_rr * rr)
            bounce = any(ry - y[f] >= p.bounce_rr * rr for f in win)
            feats.update(dx_cross_rr=dx_cross / rr, through_net=through, bounced_up=bounce,
                         size_ratio=float(size[cross] / expected_size) if np.isfinite(size[cross]) else None,
                         cross_interpolated=bool(state[cross] == 2))
            inside = abs(dx_cross) <= p.make_dx_rr * rr
            made = inside and through and not bounce
            # crude confidence: distance from the decision boundary of the main feature
            conf = float(min(1.0, abs(abs(dx_cross) / rr - p.make_dx_rr) / p.make_dx_rr + 0.3))
        shots.append(ShotCandidate(int(s), int(apex), None if cross is None else int(cross), made, conf, feats))
    return shots
