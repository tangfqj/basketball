from hoopstats.evaluation import evaluate, match_events
from hoopstats.outputs import aggregate, read_events_csv, write_events_csv
from hoopstats.schema import ShotEvent, Team, Zone


def ev(t, team="A", zone=Zone.INSIDE_ARC, made=True):
    return ShotEvent(0, t, int(t * 30), Team(team), zone, made)


def test_matching_is_one_to_one_and_closest_first():
    gt = [ev(10.0), ev(11.0)]
    pred = [ev(10.9)]
    assert match_events(gt, pred, 1.0) == [(1, 0)]


def test_evaluate():
    gt = [ev(10), ev(20, made=False), ev(30, "B", Zone.BEYOND_ARC)]
    pred = [ev(10.4), ev(20.2, made=True), ev(45)]
    r = evaluate(gt, pred, "3x3")
    assert r.n_matched == 2
    assert abs(r.attempt_recall - 2 / 3) < 1e-9
    assert abs(r.attempt_precision - 2 / 3) < 1e-9
    assert r.make_accuracy == 0.5
    assert r.points_error["B"] == 2


def test_csv_roundtrip_and_aggregate(tmp_path):
    evs = [ev(1), ev(2, "B", Zone.BEYOND_ARC), ev(3, "B", Zone.FT, made=False)]
    write_events_csv(evs, tmp_path / "e.csv")
    back = read_events_csv(tmp_path / "e.csv")
    assert [e.zone for e in back] == [e.zone for e in evs]
    s = aggregate(back, "3x3")
    assert s["total"]["points"] == 3
    assert s["teams"]["B"]["2PT"]["made"] == 1
    assert s["teams"]["B"]["FT"]["attempts"] == 1
