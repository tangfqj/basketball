import numpy as np

from hoopstats.tracking import track_ball


def synthetic(seed=0, drop=0.35):
    """A 60-frame parabolic shot + a static orange false positive + random clutter."""
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for f in range(100, 160):
        t = f - 100
        cx, cy = 1000 + 12 * t, 1500 - 40 * t + 0.9 * t * t     # up, then down (image y grows downwards)
        truth[f] = (cx, cy)
        if rng.random() > drop:                                  # missed detections
            rows.append((f, cx - 17 + rng.normal(0, 2), cy - 17 + rng.normal(0, 2), 34, 34, rng.uniform(0.3, 0.9), 0))
    for f in range(0, 200, 7):                                   # static distractor, low confidence
        rows.append((f, 3000, 400, 30, 30, 0.2, 0))
    for _ in range(15):                                          # random one-off clutter
        rows.append((rng.integers(0, 200), rng.uniform(0, 3800), rng.uniform(0, 2100), 30, 30, 0.35, 0))
    return np.array(rows, np.float32), truth


def test_tracks_parabola_and_fills_gaps():
    ball, truth = synthetic()
    out = track_ball(ball, 200)
    err = [np.hypot(out["x"][f] - x, out["y"][f] - y) for f, (x, y) in truth.items() if out["state"][f]]
    covered = sum(bool(out["state"][f]) for f in truth) / len(truth)
    assert covered > 0.9                       # gaps filled
    assert np.median(err) < 5 and max(err) < 40
    assert (out["state"][list(truth)] == 2).any()   # some frames were interpolated


def test_rejects_isolated_false_positives():
    ball, truth = synthetic(seed=3)
    out = track_ball(ball, 200)
    outside = [f for f in range(200) if f not in truth and out["state"][f]]
    # the static low-confidence distractor never starts a track; one-off clutter needs support
    assert len(outside) <= 5
