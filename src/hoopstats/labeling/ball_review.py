"""Review / correct pre-annotated ball boxes in the browser (Phase 2.2).

  hoopstats ball-review            # all videos under data/ball_frames/
  hoopstats ball-review IMG_0105   # one video

Reads data/ball_frames/<video>/{frames.json, proposals.json, images/*.jpg}; writes decisions to
labels/ball/<video>.json  ({image: {"status": "ball"|"none"|"skip", "box": [x, y, w, h]}}, 4K pixels),
saved after every decision.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

FRAMES = Path("data/ball_frames")
LABELS = Path("labels/ball")
STATIC = Path(__file__).parent / "static"


def load_labels(video: str) -> dict:
    p = LABELS / f"{video}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_labels(video: str, labels: dict) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    p = LABELS / f"{video}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(labels.items())), indent=1))
    os.replace(tmp, p)


def collect_items(videos: list[str]) -> list[dict]:
    items = []
    for v in videos:
        d = FRAMES / v
        props = json.loads((d / "proposals.json").read_text()) if (d / "proposals.json").exists() else {}
        rim = json.loads(Path(f"calib/{v}.json").read_text())["rim_center"]
        labels = load_labels(v)
        for r in json.loads((d / "frames.json").read_text()):
            name = f"{v}_f{r['frame']:05d}.jpg"
            items.append({**r, "video": v, "image": name, "proposals": props.get(name, []),
                          "label": labels.get(name), "rim": rim})
    return items


def run_ball_review(videos: list[str] | None = None, port: int = 8767, open_browser: bool = True) -> None:
    videos = videos or sorted(p.name for p in FRAMES.iterdir() if (p / "frames.json").exists())
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(self.path)
            if path in ("/", "/index.html"):
                self._send((STATIC / "ball_review.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/items":
                self._send(json.dumps(collect_items(videos)).encode(), "application/json")
            elif path.startswith("/img/"):
                _, _, video, name = path.split("/", 3)
                p = FRAMES / video / "images" / name
                if p.resolve().is_relative_to(FRAMES.resolve()) and p.exists():
                    self._send(p.read_bytes(), "image/jpeg", cache=True)
                else:
                    self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(b"not found", "text/plain", 404)
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with lock:
                labels = load_labels(d["video"])
                if d.get("label") is None:
                    labels.pop(d["image"], None)
                else:
                    labels[d["image"]] = d["label"]
                save_labels(d["video"], labels)
            self._send(b'{"ok": true}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Reviewing ball boxes for {', '.join(videos)} -> {LABELS}/\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
