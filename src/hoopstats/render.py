"""Annotated output video (requirements OUT-3): ball trail, rim, shooter box with team + bib number, shot
banners, running score.

Rendered at 1920x1080 and piped into ffmpeg (H.264), so the file plays everywhere.
"""

from __future__ import annotations

import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2

from .rules import LABELS

TEAM_COLORS = {"A": (235, 140, 40), "B": (40, 170, 250), "?": (200, 200, 200)}   # BGR
OUT_W, OUT_H = 1920, 1080


def _put(img, text, org, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_video(video, track, persons, events, shots, calibration, out_path: Path, fps: float, rules: str,
                 time_range: tuple[float, float] | None = None, trail: int = 20) -> Path:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(3))
    s = OUT_W / W
    f_start, f_end = (0, n) if time_range is None else (int(time_range[0] * fps), min(n, int(time_range[1] * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "fast",
                             "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                            stdin=subprocess.PIPE)
    labels = LABELS[rules]
    # per shot: when to show what (shot index aligned with events)
    windows = []                        # (release, decided, end of banner, event)
    for e, sh in zip(events, shots):
        decided = (sh.cross_frame or sh.apex_frame) + int(0.3 * fps)
        windows.append((sh.release_frame, decided, decided + int(1.5 * fps), e))
    shooter_boxes = {}                  # track id -> {person frame: (x, y, w, h)}
    wanted = {e.shooter_track for e in events if e.shooter_track is not None}
    for row in persons:
        if int(row[1]) in wanted:
            shooter_boxes.setdefault(int(row[1]), {})[int(row[0])] = tuple(row[2:6])
    rim_c = tuple(int(v * s) for v in calibration.rim_center)
    rim_rx = int(abs(calibration.rim_edge[1][0] - calibration.rim_edge[0][0]) / 2 * s) if calibration.rim_edge else 16
    rim_ry = int(abs(calibration.rim_edge[2][1] - calibration.rim_edge[3][1]) / 2 * s) if len(calibration.rim_edge) >= 4 else 4
    person_by_frame = {}
    for row in persons:
        person_by_frame.setdefault(int(row[0]), []).append(row)
    last_pf = None
    for f in range(f_start, f_end):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        cv2.ellipse(img, rim_c, (max(rim_rx, 6), max(rim_ry, 3)), 0, 0, 360, (0, 255, 255), 2)
        pf = f - f % 3
        rows = person_by_frame.get(pf, [])
        last_pf = rows if rows else last_pf
        active = [(e, a, d, b) for a, d, b, e in windows if a <= f <= b]
        for row in last_pf or []:
            _, _, x, y, w, h, _ = row
            cv2.rectangle(img, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), (180, 180, 180), 1)
        # ball trail
        pts = [(int(track["x"][g] * s), int(track["y"][g] * s)) for g in range(max(0, f - trail), f + 1)
               if track["state"][g] > 0]
        for p0, p1 in pairwise(pts):
            cv2.line(img, p0, p1, (0, 220, 255), 3, cv2.LINE_AA)
        if track["state"][f] > 0:
            cv2.circle(img, (int(track["x"][f] * s), int(track["y"][f] * s)), 16, (0, 220, 255), 3, cv2.LINE_AA)
        # shooter box + shot banner
        for e, a, d, b in active:
            col = TEAM_COLORS.get(e.team.value, TEAM_COLORS["?"])
            who = f"{e.team.value} #{e.player}" if e.player else f"Team {e.team.value}"
            boxes = shooter_boxes.get(e.shooter_track, {})
            near = [g for g in boxes if abs(g - f) <= 6]
            if near and f <= d:                     # follow the shooter until the result is known
                x, y, w, h = boxes[min(near, key=lambda g: abs(g - f))]
                p0, p1 = (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s))
                cv2.rectangle(img, p0, p1, col, 4)
                _put(img, who, (p0[0], max(30, p0[1] - 12)), 0.9, col, 2)
            result = ("MADE" if e.made else "MISSED") if f >= d else "shot..."
            text = f"{who}  {labels[e.zone]}  {result}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            _put(img, text, (OUT_W // 2 - tw // 2, 90), 1.4, col, 3)
        # running score (events up to their decision time)
        score = {"A": 0, "B": 0}
        att = {"A": 0, "B": 0}
        for e, (_, d, _, _) in zip(events, windows):
            if d <= f and e.team.value in score:
                score[e.team.value] += e.points
                att[e.team.value] += 1
        cv2.rectangle(img, (20, 20), (470, 130), (30, 30, 30), -1)
        _put(img, f"Team A  {score['A']:>3}   ({att['A']} att)", (35, 65), 1.0, TEAM_COLORS["A"])
        _put(img, f"Team B  {score['B']:>3}   ({att['B']} att)", (35, 110), 1.0, TEAM_COLORS["B"])
        _put(img, f"{f / fps:6.1f} s", (OUT_W - 200, OUT_H - 30), 0.9)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path
