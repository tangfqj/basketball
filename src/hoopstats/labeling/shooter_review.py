"""Review the shooter of every labeled shot in the browser (Milestone 2 step 1, requirements PGT-1).

  hoopstats shooter-review            # all videos under data/shooter_frames/
  hoopstats shooter-review IMG_0105   # one video

Reads data/shooter_frames/<video>/{items.json, *.jpg} (made by scripts/propose_shooters.py); writes
decisions to labels/shooters/<video>.json, saved after every decision:
  {"<event_id>": {"status": "ok" | "none", "track": <dataset track id> | null, "number": <bib> | null,
                  "team": "A" | "B" | null, "proposed": <track id> | null}}
"team" is the dataset's team (A = first offense group), not the A/B of the shot labels.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

FRAMES = Path("data/shooter_frames")
LABELS = Path("labels/shooters")
STATIC = Path(__file__).parent / "static"


def load_decisions(video: str, labels_dir: Path = LABELS) -> dict:
    p = labels_dir / f"{video}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_decisions(video: str, decisions: dict, labels_dir: Path = LABELS) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    p = labels_dir / f"{video}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(decisions.items(), key=lambda kv: int(kv[0]))), indent=1))
    os.replace(tmp, p)


def team_mapping(items: list[dict], players: dict) -> dict[str, str]:
    """Shot-label team letter -> dataset team letter, by majority over the proposals (letters are arbitrary)."""
    same = swap = 0
    for it in items:
        t = players.get(str(it["proposal"]), {}).get("team")
        if t and it["label_team"] in ("A", "B"):
            same += t == it["label_team"]
            swap += t != it["label_team"]
    return {"A": "A", "B": "B"} if same >= swap else {"A": "B", "B": "A"}


def flag_reason(it: dict, players: dict, mapping: dict[str, str]) -> str | None:
    """Why a proposal deserves a closer look (shown in the tool; every shot is reviewed anyway)."""
    c = it["candidates"]
    if not c:
        return "no proposal"
    t = players.get(str(c[0][0]), {}).get("team")
    if it["label_team"] in mapping and t and mapping[it["label_team"]] != t:
        return "team differs from shot label"
    if len(c) > 1 and c[1][1] - c[0][1] < 0.1:
        return "close second candidate"
    return None


def collect(videos: list[str]) -> dict:
    out = {}
    for v in videos:
        d = json.loads((FRAMES / v / "items.json").read_text())
        mapping = team_mapping(d["items"], d["players"])
        dec = load_decisions(v)
        for it in d["items"]:
            it["flag"] = flag_reason(it, d["players"], mapping)
            it["decision"] = dec.get(str(it["event_id"]))
            it["label_team_ds"] = mapping.get(it["label_team"])
        out[v] = d
    return out


def run_shooter_review(videos: list[str] | None = None, port: int = 8768, open_browser: bool = True) -> None:
    videos = videos or sorted(p.name for p in FRAMES.iterdir() if (p / "items.json").exists())
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
                self._send((STATIC / "shooter_review.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/items":
                self._send(json.dumps(collect(videos)).encode(), "application/json")
            elif path.startswith("/img/"):
                _, _, video, name = path.split("/", 3)
                p = FRAMES / video / name
                if p.resolve().is_relative_to(FRAMES.resolve()) and p.exists():
                    self._send(p.read_bytes(), "image/jpeg", cache=True)
                else:
                    self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/decision":
                return self._send(b"not found", "text/plain", 404)
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if d.get("video") not in videos:
                return self._send(b"unknown video", "text/plain", 400)
            with lock:
                dec = load_decisions(d["video"])
                if d.get("decision") is None:
                    dec.pop(str(d["event_id"]), None)
                else:
                    dec[str(d["event_id"])] = d["decision"]
                save_decisions(d["video"], dec)
            self._send(b'{"ok": true}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Reviewing shooters for {', '.join(videos)} -> {LABELS}/\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
