"""Transfer a calibration to another video from the same (slightly moved) camera.

Matches SIFT features on the static background (above the court) between reference frames and fits a
RANSAC homography, then maps the court and rim points. Writes calib/<video>.json and an overlay image to
check: data/_inspect/calib_<video>.jpg.

  uv run python scripts/derive_calibration.py IMG_0111 [--ref IMG_0104]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.court import court_lines, get_court


def frame(video: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(f"data/outdoor/{video}.MOV")
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {t} s")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ref", default="IMG_0104")
    ap.add_argument("--ref-time", type=float, default=1.0)
    ap.add_argument("--time", type=float, default=2.0)
    a = ap.parse_args()
    s = 0.5
    ref_img, img = frame(a.ref, a.ref_time), frame(a.video, a.time)
    g0 = cv2.resize(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    g1 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    mask = np.zeros_like(g0)
    mask[: int(560 * s), :] = 255                        # static background above the court
    sift = cv2.SIFT_create(4000)
    k0, d0 = sift.detectAndCompute(g0, mask)
    k1, d1 = sift.detectAndCompute(g1, mask)
    m = [x for x, y in cv2.BFMatcher().knnMatch(d0, d1, k=2) if x.distance < 0.75 * y.distance]
    p0 = np.float32([k0[x.queryIdx].pt for x in m]) / s
    p1 = np.float32([k1[x.trainIdx].pt for x in m]) / s
    H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    res = np.linalg.norm(cv2.perspectiveTransform(p0[inl.ravel() == 1].reshape(-1, 1, 2), H).reshape(-1, 2)
                         - p1[inl.ravel() == 1], axis=1)

    def tf(pts):
        return [tuple(p) for p in cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), H).reshape(-1, 2)]

    base = Calibration.load(f"calib/{a.ref}.json")
    cal = Calibration(base.court_standard, base.image_size, tf(base.image_points), base.court_points,
                      tf([base.rim_center])[0], tf(base.rim_edge),
                      f"derived from {a.ref}: background SIFT + RANSAC homography; verify visually",
                      base.landmark_names)
    cal.save(f"calib/{a.video}.json")
    court = get_court(cal.court_standard)
    for ln in court_lines(court):
        cv2.polylines(img, [cal.court_to_image(np.array(ln)).astype(np.int32)], False, (0, 0, 255), 3)
    cx, cy = (int(v) for v in cal.rim_center)
    rx = int(abs(cal.rim_edge[1][0] - cal.rim_edge[0][0]) / 2)
    ry = int(abs(cal.rim_edge[2][1] - cal.rim_edge[3][1]) / 2)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 255), 2)
    rim = cv2.resize(img[cy - 90:cy + 90, cx - 160:cx + 160], (640, 360), interpolation=cv2.INTER_NEAREST)
    full = cv2.resize(img, (1280, 720))
    Path("data/_inspect").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"data/_inspect/calib_{a.video}.jpg", np.hstack([full, np.vstack([rim, np.zeros((360, 640, 3), np.uint8)])]))
    shift = np.array(cal.rim_center) - np.array(base.rim_center)
    print(f"{a.video}: {len(m)} matches, {int(inl.sum())} inliers, residual median {np.median(res):.2f} px, "
          f"rim moved {shift.round(1)} px -> calib/{a.video}.json, check data/_inspect/calib_{a.video}.jpg")


if __name__ == "__main__":
    main()
