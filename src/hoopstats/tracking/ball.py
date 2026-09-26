"""Ball tracking: link per-frame ball detections into trajectories (Phase 2.3).

There is one game ball, so the output is one position per frame (or none):
  1. merge duplicate detections of the two passes (half-res frame + rim window);
  2. drop implausible candidates (low confidence, wrong size or shape);
  3. greedy online tracking with a constant-velocity prediction and a gate that grows with the gap;
  4. keep tracks with enough support; where tracks overlap in time, the better-supported one wins;
  5. fill short gaps inside a track by local quadratic fits (ball flight is close to a parabola).

Coordinates are 4K image pixels (ball centre).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackerParams:
    min_conf: float = 0.15          # candidates below this are ignored
    start_conf: float = 0.30        # a new track needs at least this confidence
    size_range: tuple[float, float] = (8.0, 100.0)   # ball diameter, px (4K)
    max_aspect: float = 1.7
    max_gap: int = 12               # frames a track may coast without a detection
    base_gate: float = 45.0         # px
    gap_gate: float = 20.0          # px added per missed frame
    speed_gate: float = 1.5         # x predicted displacement
    min_hits: int = 4
    min_hits_confident: int = 2     # ... or this many hits with max conf >= confident
    confident: float = 0.6


@dataclass
class Track:
    id: int
    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    size: list[float] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    v: tuple[float, float] = (0.0, 0.0)

    @property
    def score(self) -> float:
        return float(sum(self.conf))


def merge_candidates(ball: np.ndarray, p: TrackerParams) -> dict[int, np.ndarray]:
    """(N, 7) detections [frame, x, y, w, h, conf, src] -> {frame: (M, 4) [cx, cy, d, conf]}."""
    out: dict[int, list] = {}
    for f, x, y, w, h, c, _ in ball:
        d = (w + h) / 2
        if c < p.min_conf or not (p.size_range[0] <= d <= p.size_range[1]) or max(w, h) / max(1e-6, min(w, h)) > p.max_aspect:
            continue
        cx, cy = x + w / 2, y + h / 2
        lst = out.setdefault(int(f), [])
        for k, (ox, oy, od, oc) in enumerate(lst):          # duplicate of the other pass
            if np.hypot(cx - ox, cy - oy) < 0.7 * max(d, od):
                if c > oc:
                    lst[k] = (cx, cy, d, c)
                break
        else:
            lst.append((cx, cy, d, c))
    return {f: np.array(v, np.float64) for f, v in out.items()}


def build_tracks(cands: dict[int, np.ndarray], n_frames: int, p: TrackerParams) -> list[Track]:
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 0
    for f in range(n_frames):
        still = []
        for t in active:
            (finished if f - t.frames[-1] > p.max_gap else still).append(t)
        active = still
        c = cands.get(f)
        if c is None or not len(c):
            continue
        pairs = []
        for ti, t in enumerate(active):
            gap = f - t.frames[-1]
            px, py = t.xy[-1][0] + t.v[0] * gap, t.xy[-1][1] + t.v[1] * gap
            gate = p.base_gate + p.gap_gate * (gap - 1) + p.speed_gate * np.hypot(*t.v) * gap
            for ci, (cx, cy, _, cc) in enumerate(c):
                dist = np.hypot(cx - px, cy - py)
                if dist < gate:
                    pairs.append((dist / gate - 0.3 * cc, ti, ci))
        used_t, used_c = set(), set()
        for _, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t = active[ti]
            gap = f - t.frames[-1]
            cx, cy, d, cc = c[ci]
            nv = ((cx - t.xy[-1][0]) / gap, (cy - t.xy[-1][1]) / gap)
            t.v = nv if len(t.frames) == 1 else (0.5 * t.v[0] + 0.5 * nv[0], 0.5 * t.v[1] + 0.5 * nv[1])
            t.frames.append(f)
            t.xy.append((cx, cy))
            t.size.append(d)
            t.conf.append(cc)
        for ci, (cx, cy, d, cc) in enumerate(c):
            if ci not in used_c and cc >= p.start_conf:
                active.append(Track(next_id, [f], [(cx, cy)], [d], [cc]))
                next_id += 1
    finished += active
    return [t for t in finished
            if len(t.frames) >= p.min_hits or (len(t.frames) >= p.min_hits_confident and max(t.conf) >= p.confident)]


def _fill(t: Track, max_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frames, positions and 'interpolated' flags of a track with short gaps filled."""
    fr = np.array(t.frames)
    xy = np.array(t.xy)
    all_f, all_xy, interp = [fr[0]], [xy[0]], [False]
    for k in range(1, len(fr)):
        gap = fr[k] - fr[k - 1]
        if 1 < gap <= max_gap:
            lo, hi = max(0, k - 4), min(len(fr), k + 4)       # up to 4 points on each side
            deg = 2 if hi - lo >= 4 else 1
            for f in range(fr[k - 1] + 1, fr[k]):
                pos = [np.polyval(np.polyfit(fr[lo:hi], xy[lo:hi, j], deg), f) for j in (0, 1)]
                all_f.append(f)
                all_xy.append(pos)
                interp.append(True)
        all_f.append(fr[k])
        all_xy.append(xy[k])
        interp.append(False)
    return np.array(all_f), np.array(all_xy, np.float64), np.array(interp)


def track_ball(ball: np.ndarray, n_frames: int, params: TrackerParams | None = None) -> dict[str, np.ndarray]:
    """-> per-frame arrays: x, y, size, conf, track (-1 = no ball), state (0 none, 1 detected, 2 interpolated)."""
    p = params or TrackerParams()
    tracks = build_tracks(merge_candidates(ball, p), n_frames, p)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    size = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    track = np.full(n_frames, -1)
    state = np.zeros(n_frames, np.int8)
    owner_score = np.full(n_frames, -1.0)
    for t in sorted(tracks, key=lambda t: t.score):          # better tracks overwrite weaker ones
        fr, xy, interp = _fill(t, p.max_gap)
        det_conf = dict(zip(t.frames, t.conf))
        det_size = dict(zip(t.frames, t.size))
        med = float(np.median(t.size))
        for f, (px, py), it in zip(fr, xy, interp):
            if t.score > owner_score[f]:
                owner_score[f] = t.score
                x[f], y[f] = px, py
                size[f] = det_size.get(f, med)
                conf[f] = det_conf.get(f, 0.0)
                track[f] = t.id
                state[f] = 2 if it else 1
    return {"x": x, "y": y, "size": size, "conf": conf, "track": track, "state": state}
