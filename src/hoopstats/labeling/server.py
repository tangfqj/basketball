"""Tiny local web server for labeling shot events.

  hoopstats label data/outdoor/IMG_0104.MOV            -> labels/IMG_0104.csv (+ .meta.json)

The video is served with HTTP Range support so the browser can seek. Non-H.264 or > 1080p inputs are
first converted to a 720p H.264 working copy (cached) because 4K HEVC scrubs poorly in browsers;
timestamps are unchanged by the conversion. Labels are saved on every change.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..video import make_proxy, probe

LABEL_COLUMNS = ["event_id", "timestamp_s", "frame", "team", "zone", "made", "note"]
STATIC = Path(__file__).parent / "static"


def labeling_video(video: Path, cache_dir: Path) -> Path:
    info = probe(video)
    if info.codec == "h264" and info.height <= 1080:
        return video
    proxy = cache_dir / video.stem / "label_720.mp4"
    if not proxy.exists():
        print(f"Creating a 720p working copy for labeling: {proxy} (one-time, may take a few minutes)")
        make_proxy(video, proxy, height=720, crf=23)
    return proxy


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_labels(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LABEL_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for i, r in enumerate(sorted(rows, key=lambda r: float(r["timestamp_s"]))):
        w.writerow({**r, "event_id": i})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, path)  # atomic: never leaves a half-written label file


def run_label_server(video: str, out: str | None = None, cache_dir: str = "cache",
                     port: int = 8765, open_browser: bool = True) -> None:
    video_p = Path(video)
    out_p = Path(out) if out else Path("labels") / f"{video_p.stem}.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p = out_p.with_suffix(".meta.json")
    info = probe(video_p)
    play_p = labeling_video(video_p, Path(cache_dir))
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
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
                self._send((STATIC / "label.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json({"video": str(video_p), "labels": str(out_p), "fps": info.fps,
                            "n_frames": info.n_frames, "duration_s": info.duration_s})
            elif self.path == "/api/labels":
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                self._json({"events": read_labels(out_p), "meta": meta})
            elif self.path.startswith("/video"):
                self._serve_video()
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/labels":
                return self._send(b"not found", "text/plain", 404)
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with lock:
                write_labels(out_p, data.get("events", []))
                meta_p.write_text(json.dumps(data.get("meta", {}), indent=2))
            self._json({"ok": True, "n": len(data.get("events", []))})

        def _serve_video(self):
            size = play_p.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:  # suffix range: last N bytes
                    start = size - int(m.group(2))
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if m else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(play_p, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled the request while seeking

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Labeling {video_p} -> {out_p}\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
