"""Bib number of each shot's shooter (Milestone 2 step 4, requirements PL-1..PL-5).

1. Around each release, torso crops of the shooter's detected box are read by the bib classifier
   (training/bib_dataset.py preprocessing, scripts/train_bib_reader.py model) -> class probabilities.
2. Roster per video: for each team, the ROSTER_SIZE numbers with the largest confident vote mass over
   all of that team's shots (3x3: three players on court, PL-4).
3. Number per shot: confidence-weighted vote over its crops, restricted to its team's roster; no
   confident crop -> "?" (PL-5).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..training.bib_dataset import crop_torso
from ..training.bib_eval import class_number, vote

ROSTER_SIZE = 3
MIN_CONF = 0.5          # crops whose best probability is lower do not vote
MAX_OVERLAP = 0.3       # prefer views where other players overlap at most this share of the shooter's box
BEFORE, AFTER = 12, 15  # frames around the detected release (~0.4 s before the ball leaves the hands);
                        # earlier views often show another player (track switches), see
                        # doc/experiments/2026-09-27-players-dev.md


class BibReader:
    def __init__(self, weights: str, device: str | None = None, imgsz: int = 128):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="classify")
        self.device, self.imgsz = device, imgsz
        names = self.model.names
        self.classes = [names[i] for i in range(len(names))]
        self.numbers = [class_number(c) for c in self.classes]

    def read(self, crops: list[np.ndarray]) -> np.ndarray:
        """Letterboxed BGR crops -> (N, C) probabilities in self.classes order."""
        if not crops:
            return np.zeros((0, len(self.classes)), np.float32)
        res = self.model.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False)
        return np.stack([r.probs.data.cpu().numpy() for r in res]).astype(np.float32)


@dataclass
class ShotViews:
    """Torso crops of one shooter around one release (filled by the team-view callback in the pipeline)."""

    frames: list = field(default_factory=list)
    overlaps: list = field(default_factory=list)
    crops: list = field(default_factory=list)

    def add(self, frame: int, img_full: np.ndarray, box, overlap: float) -> None:
        c = crop_torso(img_full, box)
        if c is not None:
            self.frames.append(frame)
            self.overlaps.append(overlap)
            self.crops.append(c)


def select_views(frames, overlaps, release: int, before: int = BEFORE, after: int = AFTER,
                 max_overlap: float = MAX_OVERLAP) -> np.ndarray:
    """Boolean mask of the views that vote: within [release - before, release + after]; the un-overlapped
    ones if there are any, else all of them."""
    f, o = np.asarray(frames), np.asarray(overlaps)
    if not len(f):
        return np.zeros(0, bool)
    near = (f >= release - before) & (f <= release + after)
    clean = near & (o <= max_overlap)
    return clean if clean.any() else near


def team_rosters(shot_probs: list[np.ndarray], shot_teams: list[str], k: int = ROSTER_SIZE,
                 min_conf: float = MIN_CONF) -> dict[str, list[int]]:
    """team -> class indices of its k most-voted numbers (confident crops of that team's shots)."""
    mass: dict[str, np.ndarray] = {}
    for p, t in zip(shot_probs, shot_teams):
        if t == "?" or not len(p):
            continue
        conf = p[p.max(1) >= min_conf]
        if len(conf):
            mass[t] = mass.get(t, 0) + conf.sum(0)
    return {t: [int(i) for i in np.argsort(-m)[:k] if m[i] > 0] for t, m in mass.items()}


def assign_numbers(shot_probs: list[np.ndarray], shot_teams: list[str], numbers: list[int],
                   min_conf: float = MIN_CONF) -> tuple[list[str], dict[str, list[int]]]:
    """-> (player number per shot as a string, "?" if unknown; roster per team as bib numbers)."""
    rosters = team_rosters(shot_probs, shot_teams, min_conf=min_conf)
    out = []
    for p, t in zip(shot_probs, shot_teams):
        allowed = rosters.get(t) or None           # unknown team: all numbers
        k, _ = vote(p, allowed, min_conf) if len(p) else (None, 0.0)
        out.append("?" if k is None else str(numbers[k]))
    return out, {t: sorted(numbers[i] for i in r) for t, r in rosters.items()}


def players_table(events, rules_points=None) -> list[dict]:
    """Per player (team, number): attempts, makes, FG%, points (OUT-6). Events need .team, .player, .made,
    .points."""
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for e in events:
        key = (e.team.value, e.player or "?")
        a = agg[key]
        a[0] += 1
        a[1] += int(e.made)
        a[2] += e.points

    def order(k):
        return (k[0] == "?", k[0], k[1] == "?", int(k[1]) if k[1].isdigit() else 0)

    return [{"team": t, "number": n, "attempts": a, "makes": m, "fg_pct": round(m / a, 3) if a else None,
             "points": pts} for (t, n), (a, m, pts) in sorted(agg.items(), key=lambda kv: order(kv[0]))]


def assign_identities(shot_probs: list[np.ndarray], shot_teams: list[str], numbers: list[int],
                      min_conf: float = MIN_CONF) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Like assign_numbers, but the vote runs over the rosters of *both* teams and the bib also decides the
    team when its number belongs to only one roster (numbers worn in both teams keep the colour team).
    -> (team per shot, number per shot, rosters)."""
    rosters = team_rosters(shot_probs, shot_teams, min_conf=min_conf)
    union = sorted({i for r in rosters.values() for i in r}) or None
    teams, out = [], []
    for p, t in zip(shot_probs, shot_teams):
        k, _ = vote(p, union, min_conf) if len(p) else (None, 0.0)
        owners = [tm for tm, r in rosters.items() if k in r] if k is not None else []
        if len(owners) == 1:
            t = owners[0]
        teams.append(t)
        out.append("?" if k is None else str(numbers[k]))
    return teams, out, {t: sorted(numbers[i] for i in r) for t, r in rosters.items()}
