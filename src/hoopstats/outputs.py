"""Writers for events.csv and stats.json (requirements §6)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .rules import LABELS, points_for
from .schema import EVENT_COLUMNS, ShotEvent, Team, Zone


def apply_rules(events: list[ShotEvent], rules: str) -> None:
    for e in events:
        e.points = points_for(rules, e.zone, e.made)


def write_events_csv(events: list[ShotEvent], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        for e in events:
            w.writerow(e.to_row())


def read_events_csv(path: str | Path) -> list[ShotEvent]:
    """Also reads ground-truth label files (missing optional columns are allowed)."""
    out = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out.append(ShotEvent(
                event_id=int(r.get("event_id") or i),
                timestamp_s=float(r["timestamp_s"]),
                frame=int(r.get("frame") or -1),
                team=Team(r.get("team") or "?"),
                zone=Zone(r["zone"]),
                made=r["made"].strip().lower() in ("1", "true", "yes", "y"),
                points=int(r.get("points") or 0),
                court_x=float(r["court_x"]) if r.get("court_x") else None,
                court_y=float(r["court_y"]) if r.get("court_y") else None,
                confidence=float(r.get("confidence") or 1.0),
            ))
    return out


def aggregate(events: list[ShotEvent], rules: str) -> dict:
    """Per-team and total attempts / makes / FG% per zone, plus points (OUT-1)."""

    def block(evs: list[ShotEvent]) -> dict:
        res = {}
        for z in Zone:
            ze = [e for e in evs if e.zone == z]
            made = sum(e.made for e in ze)
            res[LABELS[rules][z]] = {"attempts": len(ze), "made": made,
                                     "pct": round(made / len(ze), 3) if ze else None}
        made = sum(e.made for e in evs)
        res["overall"] = {"attempts": len(evs), "made": made,
                          "pct": round(made / len(evs), 3) if evs else None}
        res["points"] = sum(points_for(rules, e.zone, e.made) for e in evs)
        return res

    teams = sorted({e.team for e in events}, key=lambda t: t.value)
    return {
        "rules": rules,
        "teams": {t.value: block([e for e in events if e.team == t]) for t in teams},
        "total": block(events),
    }


def write_stats_json(events: list[ShotEvent], rules: str, path: str | Path) -> None:
    Path(path).write_text(json.dumps(aggregate(events, rules), indent=2))
