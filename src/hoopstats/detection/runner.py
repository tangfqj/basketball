"""Run ball and player detection over a whole video and cache the results (Phase 2).

  hoopstats detect data/outdoor/IMG_0104.MOV --ball-model data/models/ball_v1.pt

Per frame (every frame): ball detection in two passes — the whole frame at half resolution (1920x1080)
and a native-resolution 1280x960 window around the rim — because the ball is ~17 px in the first and
~35 px in the second (the detector was trained on both scales). Players: COCO person detector + ByteTrack
every `person_every` frames. Results are written in chunks to cache/<video>/detections/ so an interrupted
run resumes where it stopped.

Cached arrays (float32):
  ball:    frame, x, y, w, h, conf, src (0 = half-res frame, 1 = rim window)   — 4K pixel coordinates
  persons: frame, track_id, x, y, w, h, conf                                    — 4K pixel coordinates
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..calibration import Calibration


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _class_id(model, names=("ball", "sports ball")) -> int:
    for k, v in model.names.items():
        if v in names:
            return int(k)
    raise ValueError(f"model has no ball class: {model.names}")


def chunk_path(cache: Path, c0: int) -> Path:
    return cache / "detections" / f"chunk_{c0:06d}.npz"


def load_detections(cache: str | Path) -> tuple[np.ndarray, np.ndarray]:
    parts = sorted((Path(cache) / "detections").glob("chunk_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no cached detections in {cache}/detections — run `hoopstats detect` first")
    data = [np.load(p) for p in parts]
    return (np.concatenate([d["ball"] for d in data]).reshape(-1, 7),
            np.concatenate([d["persons"] for d in data]).reshape(-1, 7))


def _frame_reader(video: str, frames: range, W: int, H: int, q: queue.Queue, stats: dict) -> None:
    """Background decoder: puts (frame, 4K image, half-resolution image) on the queue, then None."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames.start)
    for f in frames:
        tt = time.perf_counter()
        ok, img = cap.read()
        if not ok:
            break
        half = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        stats["decode"] += time.perf_counter() - tt
        q.put((f, img, half))
    q.put(None)


def run_detection(video: str, calib: str, ball_model: str, person_model: str = "data/models/yolo11s.pt",
                  cache_dir: str = "cache", start: int = 0, end: int | None = None, person_every: int = 3,
                  chunk: int = 900, ball_conf: float = 0.05, device: str | None = None, batch: int = 1) -> Path:
    """`batch` > 1 sends several frames per model call (faster on large GPUs). Decoding always runs in a
    background thread, overlapping with inference."""
    from ultralytics import YOLO

    device = device or pick_device()
    cache = Path(cache_dir) / Path(video).stem
    (cache / "detections").mkdir(parents=True, exist_ok=True)
    cal = Calibration.load(calib)
    bm, pm = YOLO(ball_model), YOLO(person_model)
    ball_cls = _class_id(bm)
    person_cls = 0
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end or n, n)
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    rx0 = int(np.clip(cal.rim_center[0] - 640, 0, W - 1280))
    ry0 = int(np.clip(cal.rim_center[1] - 600, 0, H - 960))
    # track ids must stay unique if a run is resumed in a new process (the tracker restarts at 1)
    prev = [np.load(p)["persons"] for p in (cache / "detections").glob("chunk_*.npz")]
    id_offset = int(max((a[:, 1].max() for a in prev if len(a)), default=0)) + 1 if prev else 0
    t0, done = time.time(), 0
    timing = {"decode": 0.0, "wait_for_frames": 0.0, "ball_half": 0.0, "ball_rim": 0.0, "players": 0.0}
    print(f"{video}: frames {start}-{end} on {device}, batch {batch}; ball model {ball_model}")

    def ball_pass(images, frames, src, s, dx, dy, sz, out):
        tt = time.perf_counter()
        rs = bm.predict(images, imgsz=sz, conf=ball_conf, classes=[ball_cls], device=device, verbose=False)
        timing["ball_half" if src == 0 else "ball_rim"] += time.perf_counter() - tt
        for f, r in zip(frames, rs):
            for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append((f, x0 * s + dx, y0 * s + dy, (x1 - x0) * s, (y1 - y0) * s, c, src))

    for c0 in range(start - start % chunk, end, chunk):
        out = chunk_path(cache, c0)
        c_start, c_end = max(c0, start), min(c0 + chunk, end)
        if out.exists():
            continue
        q: queue.Queue = queue.Queue(maxsize=max(4, 2 * batch))
        reader = threading.Thread(target=_frame_reader, args=(video, range(c_start, c_end), W, H, q, timing),
                                  daemon=True)
        reader.start()
        balls, persons = [], []
        finished = False
        while not finished:
            items = []
            while len(items) < batch:
                tt = time.perf_counter()
                it = q.get()
                timing["wait_for_frames"] += time.perf_counter() - tt
                if it is None:
                    finished = True
                    break
                items.append(it)
            if not items:
                break
            fs = [f for f, _, _ in items]
            ball_pass([h for _, _, h in items], fs, 0, 2.0, 0, 0, 1920, balls)
            ball_pass([img[ry0:ry0 + 960, rx0:rx0 + 1280] for _, img, _ in items], fs, 1, 1.0, rx0, ry0, 1280, balls)
            for f, _, half in items:                 # tracking is sequential by nature
                if (f - c_start) % person_every:
                    continue
                tt = time.perf_counter()
                r = pm.track(half, imgsz=1920, conf=0.25, classes=[person_cls], device=device,
                             persist=True, tracker="bytetrack.yaml", verbose=False)[0]
                ids = r.boxes.id.tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
                for (x0, y0, x1, y1), c, tid in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids):
                    persons.append((f, tid + id_offset if tid >= 0 else -1, x0 * 2, y0 * 2, (x1 - x0) * 2, (y1 - y0) * 2, c))
                timing["players"] += time.perf_counter() - tt
            done += len(items)
        reader.join()
        np.savez_compressed(out, ball=np.array(balls, np.float32).reshape(-1, 7),
                            persons=np.array(persons, np.float32).reshape(-1, 7))
        el = time.time() - t0
        print(f"  frames {c_start}-{c_end} done  ({done / el:.1f} fps, {el / 60:.1f} min)", flush=True)
        print("  time per frame: " + ", ".join(f"{k} {1000 * v / max(1, done):.0f} ms" for k, v in timing.items())
              + "  (decode runs in parallel; 'wait_for_frames' is the part it actually costs)", flush=True)
    if done:
        stats = {"frames": done, "seconds": round(time.time() - t0, 1), "fps": round(done / (time.time() - t0), 2),
                 "device": device, "batch": batch,
                 "ms_per_frame": {k: round(1000 * v / done, 1) for k, v in timing.items()}}
        (cache / "detections" / "timing.json").write_text(json.dumps(stats, indent=1))
    return cache
