"""Overlay the FIBA 3x3 court model on an Outdoor frame using the dataset keypoints (visual check)."""

import sys

import cv2
import numpy as np

from hoopstats.court import get_court
from hoopstats.datasets.trackid3x3 import outdoor_calibration

video, out = sys.argv[1], sys.argv[2]
cal = outdoor_calibration("data/trackid3x3_repo/ground_truth")
c = get_court(cal.court_standard)
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
ok, frame = cap.read()

lines = []
t = np.linspace(-1, 1, 200)
theta0 = np.arcsin((c.arc_break_y - c.hoop_y) / c.arc_radius)
ang = np.linspace(theta0, np.pi - theta0, 200)
lines.append(np.c_[c.arc_radius * np.cos(ang), c.hoop_y + c.arc_radius * np.sin(ang)])       # arc
for s in (-1, 1):
    lines.append(np.array([[s * c.corner_x, 0], [s * c.corner_x, c.arc_break_y]]))            # corners
    lines.append(np.array([[s * c.lane_width / 2, 0], [s * c.lane_width / 2, c.ft_line_y]]))  # lane
lines.append(np.array([[-c.lane_width / 2, c.ft_line_y], [c.lane_width / 2, c.ft_line_y]]))   # FT line
lines.append(np.c_[t * c.width / 2, np.zeros_like(t)])                                         # baseline
for ln in lines:
    px = cal.court_to_image(ln).astype(np.int32)
    cv2.polylines(frame, [px], False, (0, 0, 255), 6)
hoop = cal.court_to_image(np.array([[0, c.hoop_y]]))[0].astype(int)
cv2.circle(frame, tuple(hoop), 12, (0, 255, 255), -1)
print("reprojection error (m):", cal.reprojection_error())
cv2.imwrite(out, cv2.resize(frame, (1600, 900)))
