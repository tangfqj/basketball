import numpy as np

from hoopstats.events import detect_shots

RIM, RR, FPS = (1800.0, 840.0), 33.0, 30.0


def make_track(n, points):
    """points: {frame: (x, y)} -> tracker-style dict with 'detected' state."""
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    size, state = np.full(n, 35.0), np.zeros(n, np.int8)
    for f, (px, py) in points.items():
        x[f], y[f], state[f] = px, py, 1
    return {"x": x, "y": y, "size": size, "state": state, "conf": np.ones(n), "track": np.zeros(n, int)}


def flight(f0, x0, y0, x1, peak_above, n_up=25, n_down=30, end_dy=400, ax=None):
    """Release at (x0, y0), apex `peak_above` px above the rim at x between shooter and x1, then down to x1."""
    pts = {}
    ax = (x0 + x1) / 2 if ax is None else ax
    ay = RIM[1] - peak_above
    for k in range(n_up + 1):
        t = k / n_up
        pts[f0 + k] = (x0 + (ax - x0) * t, y0 + (ay - y0) * (1 - (1 - t) ** 2))
    for k in range(1, n_down + 1):
        t = k / n_down
        pts[f0 + n_up + k] = (ax + (x1 - ax) * t, ay + (RIM[1] + end_dy - ay) * t * t)
    return pts


def test_make_miss_and_rim_bounce():
    pts = {}
    pts.update(flight(100, 1750, 1300, RIM[0] + 2, 180))            # swish: straight down through the rim
    pts.update(flight(300, 1500, 1250, RIM[0] + 260, 90, ax=RIM[0] + 120))   # passes beside the rim
    # rim bounce after the second shot: rises from rim level -> must not count as a new shot
    for k in range(15):
        pts[360 + k] = (RIM[0] + 150 + 3 * k, RIM[1] - 8 * k)
    shots = detect_shots(make_track(500, pts), RIM, RR, FPS)
    assert [s.release_frame for s in shots] == [100, 300]
    assert [s.made for s in shots] == [True, False]


def test_rise_that_stays_below_rim_is_not_a_shot():
    pts = flight(50, 1700, 1500, 1750, peak_above=-120)             # apex 120 px *below* the rim: a pass
    assert detect_shots(make_track(200, pts), RIM, RR, FPS) == []
