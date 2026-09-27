import numpy as np

from hoopstats.training.bib_dataset import letterbox, occlusion, torso_box
from hoopstats.training.bib_eval import class_name, class_number, evaluate, vote


def test_torso_box_and_occlusion():
    x, y, w, h = torso_box(100, 200, 50, 100)
    assert (x, w) == (100, 50) and abs(y - 215) < 1e-9 and abs(h - 40) < 1e-9
    assert occlusion((0, 0, 10, 10), []) == 0.0
    assert occlusion((0, 0, 10, 10), [(0, 0, 5, 10)]) == 0.5
    assert occlusion((0, 0, 10, 10), [(-5, -5, 30, 30)]) == 1.0


def test_letterbox_keeps_aspect():
    out = letterbox(np.zeros((40, 80, 3), np.uint8), 128)
    assert out.shape == (128, 128, 3)
    assert out[0, 0, 0] == 114 and out[64, 64, 0] == 0


def test_class_names():
    assert class_name(4) == "b04" and class_number("b12") == 12
    assert sorted([class_name(n) for n in (10, 4, 1)]) == ["b01", "b04", "b10"]


def test_vote_roster_and_confidence():
    p = np.array([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.4, 0.35, 0.25]])
    assert vote(p)[0] == 2                       # crop 3 (max 0.4) does not vote; 0.9 for class 2 vs 0.7
    assert vote(p, allowed=[0, 1])[0] == 0       # roster without class 2: renormalised, every crop votes
    assert vote(p, min_conf=0.99) == (None, 0.0)
    assert vote(np.zeros((0, 3))) == (None, 0.0)


def test_evaluate_counts():
    classes = ["b04", "b10"]
    rows = [{"video": "V", "frame": f, "track": 1, "team": "A", "number": 4, "occl": 0.0} for f in (0, 8)]
    rows += [{"video": "V", "frame": 0, "track": 2, "team": "A", "number": 10, "occl": 0.5}]
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    m = evaluate(rows, probs, classes)
    assert m["crops_clean"] == 2 and m["crop_acc_clean"] == 0.5
    assert m["windows"] == 2 and m["window_acc_roster"] == 1.0


def _p(*rows):
    return np.array(rows, dtype=float)


def test_rosters_and_numbers():
    from hoopstats.players.bib import assign_identities, assign_numbers, team_rosters

    numbers = [4, 6, 9, 10]
    probs = [_p([0.9, 0.1, 0, 0]), _p([0, 0.95, 0.05, 0]), _p([0, 0, 0.1, 0.9]), _p([0.6, 0, 0, 0.4]),
             _p([0.2, 0.2, 0.3, 0.3])]
    teams = ["A", "A", "B", "B", "A"]
    r = team_rosters(probs, teams, k=2)
    assert sorted(r["A"]) == [0, 1] and sorted(r["B"]) == [0, 3]
    nums, rosters = assign_numbers(probs, teams, numbers)
    assert nums == ["4", "6", "10", "4", "?"]     # shot 5: no crop >= 0.5 within roster A {4, 6, 9}
    assert rosters["A"] == [4, 6, 9]
    t, n, _ = assign_identities([_p([0, 0, 0.1, 0.9])], ["A"], numbers)
    assert (t, n) == (["A"], ["10"])                                  # single team: nothing to correct
    assert assign_numbers([np.zeros((0, 4))], ["A"], numbers)[0] == ["?"]


def test_select_views():
    from hoopstats.players.bib import select_views

    frames, ov = [88, 94, 100, 106, 112, 118], [0.0, 0.0, 0.5, 0.1, 0.9, 0.0]
    assert select_views(frames, ov, 100, before=12, after=15, max_overlap=0.3).tolist() == [
        True, True, False, True, False, False]
    assert select_views(frames, ov, 100, before=0, after=12, max_overlap=0.05).tolist() == [
        False, False, True, True, True, False]                        # no clean view: all views in the window
    assert select_views([], [], 100).tolist() == []


def test_players_table():
    from hoopstats.players.bib import players_table
    from hoopstats.schema import ShotEvent, Team, Zone

    ev = [ShotEvent(0, 1, 30, Team.A, Zone.INSIDE_ARC, True, points=1, player="10"),
          ShotEvent(1, 2, 60, Team.A, Zone.BEYOND_ARC, False, player="10"),
          ShotEvent(2, 3, 90, Team.UNKNOWN, Zone.INSIDE_ARC, True, points=1, player="4"),
          ShotEvent(3, 4, 120, Team.A, Zone.INSIDE_ARC, True, points=1, player="?")]
    t = players_table(ev)
    assert [(r["team"], r["number"]) for r in t] == [("A", "10"), ("A", "?"), ("?", "4")]
    assert t[0] == {"team": "A", "number": "10", "attempts": 2, "makes": 1, "fg_pct": 0.5, "points": 1}
