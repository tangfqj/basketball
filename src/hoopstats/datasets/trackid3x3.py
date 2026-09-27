"""Loaders for the TrackID3x3 ground truth (https://github.com/open-starlab/TrackID3x3, CC BY 4.0).

Expected local layout (git-ignored):  data/trackid3x3_repo/ground_truth/{Outdoor,Indoor}/...

Available annotations (Outdoor):
  * MOT/<video>.txt                    player boxes, all frames, MOTChallenge format, 4K pixels
  * delimitation_frames/<video>.csv    game-flow events (start, out of bounds, FG success/failure,
                                       resume...) with jersey numbers and offense/defense player ids
  * court_keypoints/Outdoor.ndjson     4 court-corner keypoints (Labelbox export)
  * pose/IMG_0104_.json                10 body keypoints, first 3600 frames of IMG_0104 only
  * MOT/split_transformed/             court-coordinate tracks split into possession segments
Note: the delimitation files are NOT complete shot labels (only shots that end a segment are listed).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..calibration import Calibration
from ..detection import Detection

# Court positions of the Outdoor keypoints, from the dataset's own scripts (centimetres,
# x along the baseline starting at the right corner, y from the baseline).
OUTDOOR_KEYPOINTS_CM = {"rku_0": (0, 0), "rku_1": (0, 1105), "rku_10": (1505, 0), "rku_11": (1505, 1105)}
OUTDOOR_COURT_WIDTH_CM = 1505


def outdoor_cm_to_court(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Dataset court coordinates (cm) -> hoopstats court coordinates (m, origin at baseline midpoint)."""
    return ((OUTDOOR_COURT_WIDTH_CM / 2 - x_cm) / 100.0, y_cm / 100.0)


def load_mot(path: str | Path) -> list[Detection]:
    dets = []
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            dets.append(Detection(frame=int(float(p[0])), track_id=int(float(p[1])),
                                  x=float(p[2]), y=float(p[3]), w=float(p[4]), h=float(p[5]),
                                  score=float(p[6]) if len(p) > 6 else 1.0, cls="person"))
    return dets


@dataclass
class FlowEvent:
    frame: int
    label: str                                         # e.g. "FG success", "Out of bounds"
    jersey: dict[int, int] = field(default_factory=dict)   # track id -> jersey number
    offense: list[int] = field(default_factory=list)       # track ids
    defense: list[int] = field(default_factory=list)


_HEAD = re.compile(r"^\s*(\d+)\s*\((.*)\)\s*$")


def _ids(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def load_delimitation(path: str | Path) -> list[FlowEvent]:
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not (m := _HEAD.match(row[0])):
                continue
            ev = FlowEvent(frame=int(m.group(1)), label=" ".join(m.group(2).split()))
            if len(row) > 1 and row[1].strip():
                # "id: jersey" pairs; tolerant of separator typos in the dataset (e.g. "5: 6. 6: 6")
                for k, v in re.findall(r"(\d+)\s*:\s*(\d+)", row[1]):
                    ev.jersey[int(k)] = int(v)
            if len(row) > 2 and row[2].strip():
                off, _, de = row[2].partition(";")
                ev.offense, ev.defense = _ids(off), _ids(de)
            out.append(ev)
    return out


def team_membership(events: list[FlowEvent]) -> dict[int, str]:
    """Track id -> team "A"/"B" (A = the first offense group listed)."""
    first = next((e for e in events if e.offense), None)
    if first is None:
        return {}
    return {**{i: "A" for i in first.offense}, **{i: "B" for i in first.defense}}


def load_court_keypoints(path: str | Path) -> dict[str, tuple[float, float]]:
    """Labelbox ndjson -> {keypoint value: (x, y) pixels} for the first labelled image."""
    row = json.loads(Path(path).read_text().splitlines()[0])
    label = next(iter(row["projects"].values()))["labels"][0]
    return {o["value"]: (o["point"]["x"], o["point"]["y"]) for o in label["annotations"]["objects"]}


def outdoor_calibration(gt_root: str | Path) -> Calibration:
    """Calibration built from the dataset's court keypoints (requirements CAL-6). No rim yet."""
    kp = load_court_keypoints(Path(gt_root) / "Outdoor" / "court_keypoints" / "Outdoor.ndjson")
    names = [n for n in OUTDOOR_KEYPOINTS_CM if n in kp]
    return Calibration(
        court_standard="fiba_3x3",
        image_size=(3840, 2160),
        image_points=[kp[n] for n in names],
        court_points=[outdoor_cm_to_court(*OUTDOOR_KEYPOINTS_CM[n]) for n in names],
        note="from TrackID3x3 Outdoor court keypoints",
    )
