"""Browser-based calibration tool (Phase 1, requirements CAL-1..CAL-5).

  hoopstats calibrate data/outdoor/IMG_0104.MOV --init trackid3x3-outdoor   -> calib/IMG_0104.json
  hoopstats calibrate data/outdoor/IMG_0105.MOV --init calib/IMG_0104.json  (same camera setup, CAL-4)

Click >= 4 court landmarks and 4 points on the rim (left, right, front, back) on a full-resolution
reference frame; the court model is drawn over the frame so the fit can be checked before saving.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..court import court_landmarks, court_lines, get_court
from .homography import Calibration

STATIC = Path(__file__).parent / "static"
RIM_POINTS = ["rim_left", "rim_right", "rim_front", "rim_back"]


def extract_frame(video: Path, t: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(out)], check=True)
    return out


def _initial_points(init: str | None, court_name: str) -> tuple[dict, dict, str]:
    """-> (points {name: [px, py]}, extra landmarks {name: [cx, cy]}, court standard)."""
    if not init:
        return {}, {}, court_name
    if init == "trackid3x3-outdoor":
        from ..datasets.trackid3x3 import outdoor_calibration

        cal = outdoor_calibration("data/trackid3x3_repo/ground_truth")
        names = ["trackid_" + n for n in ("rku_0", "rku_1", "rku_10", "rku_11")]
    else:
        cal = Calibration.load(init)
        names = cal.landmark_names or [f"point_{i}" for i in range(len(cal.image_points))]
    known = court_landmarks(get_court(cal.court_standard))
    points = {n: list(p) for n, p in zip(names, cal.image_points)}
    extra = {n: list(c) for n, c in zip(names, cal.court_points) if n not in known}
    for name, p in zip(RIM_POINTS, cal.rim_edge):
        points[name] = list(p)
    return points, extra, cal.court_standard


def build_calibration(court_name: str, image_size, points: dict, landmarks: dict) -> Calibration:
    names = [n for n in points if n in landmarks]
    rim = [points[n] for n in RIM_POINTS if n in points]
    center = None
    if "rim_left" in points and "rim_right" in points:
        cx = (points["rim_left"][0] + points["rim_right"][0]) / 2
        ys = [points[n][1] for n in ("rim_front", "rim_back") if n in points] or \
             [(points["rim_left"][1] + points["rim_right"][1]) / 2]
        center = (cx, sum(ys) / len(ys))
    return Calibration(court_name, tuple(image_size), [tuple(points[n]) for n in names],
                       [tuple(landmarks[n]) for n in names], center, [tuple(p) for p in rim],
                       note="hoopstats calibrate", landmark_names=names)


def run_calibration_tool(video_path: str, out_path: str | None = None, court_standard: str = "fiba_3x3",
                         init: str | None = None, t: float = 1.0, cache_dir: str = "cache",
                         port: int = 8766, open_browser: bool = True) -> None:
    video = Path(video_path)
    out = Path(out_path) if out_path else Path("calib") / f"{video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = extract_frame(video, t, Path(cache_dir) / video.stem / f"calib_frame_{t:g}s.jpg")
    import cv2

    h, w = cv2.imread(str(frame)).shape[:2]
    points, extra, court_standard = _initial_points(init or (str(out) if out.exists() else None), court_standard)
    court = get_court(court_standard)
    landmarks = {**court_landmarks(court), **extra}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send((STATIC / "calibrate.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/frame.jpg":
                self._send(frame.read_bytes(), "image/jpeg")
            elif self.path == "/api/init":
                self._json({"video": str(video), "out": str(out), "image_size": [w, h],
                            "court_standard": court_standard,
                            "landmarks": [{"name": n, "court": list(c)} for n, c in landmarks.items()],
                            "rim_points": RIM_POINTS, "points": points})
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pts = {k: v for k, v in data.get("points", {}).items() if v}
            n_court = sum(1 for k in pts if k in landmarks)
            if self.path == "/api/preview":
                if n_court < 4:
                    return self._json({"lines": [], "error_m": None})
                cal = build_calibration(court_standard, (w, h), pts, landmarks)
                lines = [cal.court_to_image(np.array(ln)).round(1).tolist() for ln in court_lines(court)]
                return self._json({"lines": lines, "error_m": cal.reprojection_error()})
            if self.path == "/api/save":
                if n_court < 4:
                    return self._json({"ok": False, "error": "need >= 4 court landmarks"}, 400)
                build_calibration(court_standard, (w, h), pts, landmarks).save(out)
                return self._json({"ok": True, "path": str(out)})
            self._send(b"not found", "text/plain", 404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Calibrating {video} -> {out}\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
