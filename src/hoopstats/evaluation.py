"""Event-level evaluation against hand labels (requirements §7.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import points_for
from .schema import ShotEvent, Team


@dataclass
class EvalResult:
    n_gt: int
    n_pred: int
    n_matched: int
    attempt_recall: float | None
    attempt_precision: float | None
    make_accuracy: float | None
    zone_accuracy: float | None
    team_accuracy: float | None
    points_error: dict[str, int] = field(default_factory=dict)
    matches: list[tuple[int, int]] = field(default_factory=list)   # (gt index, pred index)


def match_events(gt: list[ShotEvent], pred: list[ShotEvent], tol_s: float = 1.0) -> list[tuple[int, int]]:
    """One-to-one matching by release time: closest pairs first, within +-tol_s."""
    cands = sorted(
        (abs(g.timestamp_s - p.timestamp_s), i, j)
        for i, g in enumerate(gt) for j, p in enumerate(pred)
        if abs(g.timestamp_s - p.timestamp_s) <= tol_s
    )
    used_g, used_p, out = set(), set(), []
    for _, i, j in cands:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return sorted(out)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def evaluate(gt: list[ShotEvent], pred: list[ShotEvent], rules: str, tol_s: float = 1.0) -> EvalResult:
    m = match_events(gt, pred, tol_s)
    pairs = [(gt[i], pred[j]) for i, j in m]
    team_pairs = [(g, p) for g, p in pairs if g.team != Team.UNKNOWN]

    def pts(evs: list[ShotEvent], team: Team | None) -> int:
        return sum(points_for(rules, e.zone, e.made) for e in evs if team is None or e.team == team)

    teams = sorted({e.team for e in gt if e.team != Team.UNKNOWN}, key=lambda t: t.value)
    points_error = {t.value: abs(pts(pred, t) - pts(gt, t)) for t in teams}
    points_error["total"] = abs(pts(pred, None) - pts(gt, None))

    return EvalResult(
        n_gt=len(gt), n_pred=len(pred), n_matched=len(m),
        attempt_recall=_ratio(len(m), len(gt)),
        attempt_precision=_ratio(len(m), len(pred)),
        make_accuracy=_ratio(sum(g.made == p.made for g, p in pairs), len(pairs)),
        zone_accuracy=_ratio(sum(g.zone == p.zone for g, p in pairs), len(pairs)),
        team_accuracy=_ratio(sum(g.team == p.team for g, p in team_pairs), len(team_pairs)),
        points_error=points_error,
        matches=m,
    )


def format_report(r: EvalResult) -> str:
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    return "\n".join([
        f"GT attempts: {r.n_gt}   predicted: {r.n_pred}   matched: {r.n_matched}",
        f"attempt recall     {f(r.attempt_recall)}",
        f"attempt precision  {f(r.attempt_precision)}",
        f"make/miss accuracy {f(r.make_accuracy)}",
        f"zone accuracy      {f(r.zone_accuracy)}",
        f"team accuracy      {f(r.team_accuracy)}",
        f"points error       {r.points_error}",
    ])
