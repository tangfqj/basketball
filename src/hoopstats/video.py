"""Video probing and working-copy (proxy) creation via ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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


def _hwaccel() -> list[str]:
    """Hardware HEVC/H.264 decoding on macOS (VideoToolbox); software elsewhere."""
    return ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []


def _passthrough_flag() -> list[str]:
    """ffmpeg >= 5.1 uses -fps_mode; older versions only know -vsync."""
    opts = subprocess.run([_require("ffmpeg"), "-hide_banner", "-h", "full"],
                          capture_output=True, text=True, check=False).stdout
    return ["-fps_mode", "passthrough"] if "-fps_mode" in opts else ["-vsync", "passthrough"]


def make_proxy(src: str | Path, dst: str | Path, height: int = 1080, crf: int = 18) -> Path:
    """Transcode to an H.264 working copy (fast to decode/seek); frame timestamps are preserved."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial" + dst.suffix)
    subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", *_hwaccel(), "-i", str(src),
         "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
         "-g", "30",  # frequent keyframes: fast, accurate seeking
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", *_passthrough_flag(), str(tmp)],
        check=True,
    )
    tmp.replace(dst)  # only a finished file ever appears at dst
    return dst
