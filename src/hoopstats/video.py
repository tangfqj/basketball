"""Video probing and working-copy (proxy) creation via ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"{tool} not found on PATH (install ffmpeg, see README)")
    return exe


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    s, fmt = d["streams"][0], d["format"]
    fps = float(Fraction(s["r_frame_rate"]))
    duration = float(fmt["duration"])
    n = int(s.get("nb_frames") or round(duration * fps))
    return VideoInfo(str(path), int(s["width"]), int(s["height"]), fps, n, duration, s["codec_name"])


def make_proxy(src: str | Path, dst: str | Path, height: int = 1080, crf: int = 18) -> Path:
    """Transcode to an H.264 working copy (fast to decode), keeping the frame rate and frame count."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-i", str(src),
         "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
         "-an", "-fps_mode", "passthrough", str(dst)],
        check=True,
    )
    return dst
