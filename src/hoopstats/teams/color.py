"""Team assignment from jersey / bib colour (Phase 5, requirements TM-1..TM-4).

Per video, without labels:
  1. crop the torso of every on-court player detection (people off court are removed beforehand with the
     calibration, TM-4);
  2. describe it by a hue/saturation histogram of its *saturated* pixels — bibs are strongly coloured,
     while skin, shorts, shadows and the court are mostly excluded by the saturation/value gates;
  3. k-means with k = 2 over all detections of the video;
  4. detections with too few coloured pixels (e.g. a referee in black) or far from both centres are
     "unknown"; a track's team is the majority vote of its detections.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

H_BINS, S_BINS = 18, 4
MIN_SAT, MIN_VAL = 60, 50


def torso_crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Central torso region of a person box (x, y, w, h in image pixels)."""
    x, y, w, h = box
    x0, x1 = int(x + 0.25 * w), int(x + 0.75 * w)
    y0, y1 = int(y + 0.15 * h), int(y + 0.50 * h)
    H, W = img.shape[:2]
    return img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]


def estimate_court_hue(img: np.ndarray, calibration, court, n: int = 400, seed: int = 0) -> float | None:
    """Dominant hue of the (saturated) court surface, sampled at random floor points inside the court.
    Pixels of this hue are ignored in torso crops: court visible behind a player is not a bib colour."""
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(-court.width / 2, court.width / 2, n), rng.uniform(0, court.half_length, n)]
    px = calibration.court_to_image(pts).round().astype(int)
    H, W = img.shape[:2]
    px = px[(px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)]
    hsv = cv2.cvtColor(img[px[:, 1], px[:, 0]].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[hsv[:, 1] >= MIN_SAT]
    if len(sat) < 0.3 * len(hsv):      # unsaturated court (e.g. wood, grey): nothing to exclude
        return None
    return float(np.median(sat[:, 0]))


def _hue_dist(h: np.ndarray, ref: float) -> np.ndarray:
    d = np.abs(h.astype(np.int16) - ref)
    return np.minimum(d, 180 - d)


def color_feature(crop: np.ndarray, background_hue: float | None = None,
                  hue_tol: float = 12) -> tuple[np.ndarray | None, float]:
    """-> (L1-normalised, square-rooted H×S histogram of saturated pixels, fraction of saturated pixels).
    Pixels within `hue_tol` of `background_hue` (the court surface) are ignored."""
    if crop.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = (hsv[..., 1] >= MIN_SAT) & (hsv[..., 2] >= MIN_VAL)
    if background_hue is not None:
        keep &= _hue_dist(hsv[..., 0], background_hue) > hue_tol
    mask = keep.astype(np.uint8)
    frac = float(mask.mean())
    if mask.sum() < 20:
        return None, frac
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, MIN_SAT, 256]).ravel()
    hist /= hist.sum()
    return np.sqrt(hist).astype(np.float32), frac   # Hellinger embedding: Euclidean ~ Bhattacharyya


@dataclass
class TeamModel:
    centers: np.ndarray          # (2, D)
    reject_dist: float           # distances above this -> unknown
    min_sat_frac: float = 0.15

    def predict(self, feat: np.ndarray | None, sat_frac: float) -> str:
        if feat is None or sat_frac < self.min_sat_frac:
            return "?"
        d = np.linalg.norm(self.centers - feat, axis=1)
        k = int(np.argmin(d))
        return "?" if d[k] > self.reject_dist else "AB"[k]


def fit_team_model(features: list[np.ndarray], min_sat_frac: float = 0.15, reject_pct: float = 97.5,
                   seed: int = 0) -> TeamModel:
    X = np.stack(features).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, labels, centers = cv2.kmeans(X, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    d = np.linalg.norm(X - centers[labels.ravel()], axis=1)
    # deterministic naming: team A = the cluster whose centre has the lower dominant hue bin
    order = np.argsort([np.argmax(c.reshape(H_BINS, S_BINS).sum(1)) for c in centers])
    return TeamModel(centers[order], float(np.percentile(d, reject_pct)) * 1.25, min_sat_frac)


def vote(labels: list[str], min_share: float = 0.6) -> str:
    """Majority team of a track; '?' if no clear majority."""
    c = Counter(lab for lab in labels if lab != "?")
    if not c:
        return "?"
    team, n = c.most_common(1)[0]
    return team if n / sum(c.values()) >= min_share else "?"
