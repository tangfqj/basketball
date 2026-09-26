"""Build the YOLO ball dataset (ball_dataset.zip) from reviewed ball labels (Phase 2.2).

Every reviewed frame ("ball" or "none"; "skip" is left out) gives two training images:
  * native: 1280x1280 crop of the 4K frame (ball ~35 px)   — like the rim region at inference;
  * half  : 1280x1080 crop of the frame downscaled to 1920x1080 (ball ~17 px) — like a whole-frame pass.
Both fit imgsz=1280 without further resizing, so the model learns both ball sizes at their real pixel
scale. Crops containing a ball are placed at a random offset (the ball is never at a fixed position);
"none" frames give background-only images (hard negatives). Train/val follows frames.json (time split).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import zipfile
from pathlib import Path

import cv2

FRAMES = Path("data/ball_frames")


def _crop_origin(W, H, cw, ch, box, rng, margin=48, fallback=None):
    """Top-left of a cw x ch crop containing `box` (with margin) at a random position; else around fallback."""
    if box is not None:
        x, y, w, h = box
        for m in (margin, 0):            # near the image border the margin may not fit
            lo_x, hi_x = math.ceil(max(0, x + w + m - cw)), math.floor(min(W - cw, x - m))
            lo_y, hi_y = math.ceil(max(0, y + h + m - ch)), math.floor(min(H - ch, y - m))
            if lo_x <= hi_x and lo_y <= hi_y:
                return rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
        raise ValueError(f"box {box} does not fit in a {cw}x{ch} crop of a {W}x{H} image")
    cx, cy = fallback if fallback is not None else (rng.uniform(0, W), rng.uniform(0, H))
    return (int(min(max(0, cx - cw / 2 + rng.uniform(-cw / 4, cw / 4)), W - cw)),
            int(min(max(0, cy - ch / 2 + rng.uniform(-ch / 4, ch / 4)), H - ch)))


def _write(img, box, ox, oy, cw, ch, stem, split, root):
    crop = img[oy:oy + ch, ox:ox + cw]
    cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    line = ""
    if box is not None:   # clip to the crop (a box can touch the image border)
        x0, y0 = max(0.0, box[0] - ox), max(0.0, box[1] - oy)
        x1, y1 = min(cw, box[0] + box[2] - ox), min(ch, box[1] + box[3] - oy)
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        line = f"0 {(x + w / 2) / cw:.6f} {(y + h / 2) / ch:.6f} {w / cw:.6f} {h / ch:.6f}\n"
    (root / "labels" / split / f"{stem}.txt").write_text(line)


def build_ball_dataset(out: str = "data/ball_dataset", labels_dir: str = "labels/ball", seed: int = 0) -> dict:
    rng = random.Random(seed)
    root = Path(out)
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True)
    stats = {"train": {"frames": 0, "balls": 0, "images": 0}, "val": {"frames": 0, "balls": 0, "images": 0}}
    for lab_p in sorted(Path(labels_dir).glob("*.json")):
        video = lab_p.stem
        labels = json.loads(lab_p.read_text())
        frames = {f"{video}_f{r['frame']:05d}.jpg": r for r in json.loads((FRAMES / video / "frames.json").read_text())}
        rim = json.loads(Path(f"calib/{video}.json").read_text())["rim_center"]
        for name, lab in sorted(labels.items()):
            if lab.get("status") not in ("ball", "none") or name not in frames:
                continue
            split = frames[name]["split"]
            img = cv2.imread(str(FRAMES / video / "images" / name))
            H, W = img.shape[:2]
            box = lab.get("box") if lab["status"] == "ball" else None
            stem = Path(name).stem
            # native-resolution crop (negatives: around the rim half of the time — where false alarms matter)
            fb = rim if (box is None and rng.random() < 0.5) else None
            ox, oy = _crop_origin(W, H, 1280, 1280, box, rng, fallback=fb)
            _write(img, box, ox, oy, 1280, 1280, stem + "_native", split, root)
            # half-resolution crop
            half = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            hbox = [v / 2 for v in box] if box is not None else None
            ox, oy = _crop_origin(W // 2, H // 2, 1280, 1080, hbox, rng, margin=24)
            _write(half, hbox, ox, oy, 1280, 1080, stem + "_half", split, root)
            stats[split]["frames"] += 1
            stats[split]["balls"] += box is not None
            stats[split]["images"] += 2
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    (root / "stats.json").write_text(json.dumps(stats, indent=1))
    zpath = root.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # JPEGs are already compressed
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, Path(root.name) / p.relative_to(root))
    stats["zip"] = str(zpath)
    stats["zip_mb"] = round(zpath.stat().st_size / 1e6, 1)
    return stats
