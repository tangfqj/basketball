"""Learned make/miss classifier on trajectory features (Phase 3.2).

Small logistic regression (numpy, L2) over features of the rim-level crossing computed in events/shots.py.
Shots without a downward rim-level crossing near the basket are misses by rule and never reach the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = [
    "abs_dx_cross_rr",   # horizontal distance from the rim centre at the crossing (rim radii)
    "dvy",               # vertical speed after - before the crossing (px/frame): net braking < 0, free fall > 0
    "vy_in",             # vertical speed before the crossing
    "vx_keep",           # |vx after| / |vx before| (clipped): horizontal braking
    "apex_dy",           # apex height relative to the rim (px, negative = above)
    "max_dy_after_rr",   # how far below the rim the ball gets within 0.5 s (rim radii)
    "bounced_up",        # came back above the rim after crossing
    "under_rim",         # reached net depth under the rim
    "size_ratio",        # apparent ball size / expected size at rim depth (depth cue)
]


def feature_vector(f: dict) -> np.ndarray | None:
    if f.get("crossing") == "none" or f.get("dx_cross_rr") is None:
        return None

    def g(k, default=0.0):
        v = f.get(k)
        return default if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

    vx_in, vx_out = g("vx_in"), g("vx_out")
    vy_in, vy_out = g("vy_in"), g("vy_out", g("vy_in"))
    return np.array([
        abs(g("dx_cross_rr")), vy_out - vy_in, vy_in,
        min(2.0, abs(vx_out) / max(2.0, abs(vx_in))),
        g("apex_dy"), g("max_dy_after_rr", 3.0), float(bool(f.get("bounced_up"))),
        float(bool(f.get("under_rim"))), g("size_ratio", 1.0),
    ])


@dataclass
class MakeModel:
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float
    threshold: float = 0.5

    def prob(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.std) @ self.w + self.b
        return 1 / (1 + np.exp(-z))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                                          "w": self.w.tolist(), "b": self.b, "threshold": self.threshold}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> MakeModel:
        d = json.loads(Path(path).read_text())
        if d["features"] != FEATURES:
            raise ValueError("feature list changed since this model was trained — retrain it")
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), d["b"], d["threshold"])


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 3000, lr: float = 0.1) -> MakeModel:
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float(np.mean(p - y))
    return MakeModel(mean, std, w, b)
