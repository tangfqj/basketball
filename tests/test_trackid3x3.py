"""Runs only when the TrackID3x3 ground truth is present under data/ (git-ignored)."""

from pathlib import Path

import pytest

from hoopstats.court import get_court, is_on_court
from hoopstats.datasets import trackid3x3 as t

GT = Path(__file__).resolve().parents[1] / "data" / "trackid3x3_repo" / "ground_truth"
needs_gt = pytest.mark.skipif(not GT.exists(), reason="TrackID3x3 ground truth not downloaded")


@needs_gt
def test_mot_and_delimitation():
    dets = t.load_mot(GT / "Outdoor" / "MOT" / "IMG_0104.txt")
    assert len(dets) > 10_000
    flow = t.load_delimitation(GT / "Outdoor" / "delimitation_frames" / "IMG_0104.csv")
    assert flow[0].label == "Game start"
    teams = t.team_membership(flow)
    assert sorted(teams) == sorted({d.track_id for d in dets})


@needs_gt
def test_outdoor_calibration_maps_players_onto_court():
    cal = t.outdoor_calibration(GT)
    court = get_court(cal.court_standard)
    dets = [d for d in t.load_mot(GT / "Outdoor" / "MOT" / "IMG_0104.txt") if d.frame == 200]
    pts = cal.image_to_court([d.foot_point for d in dets])
    assert all(is_on_court(court, x, y, margin=1.0) for x, y in pts)


def test_delimitation_tolerates_separator_typos(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text('168 (Game start),"1: 9, 2: 9, 4: 10, 5: 6. 6: 6, 7: 11","Offense: 2, 6, 7; Defense: 1, 4, 5"\n')
    ev = t.load_delimitation(p)[0]
    assert ev.jersey == {1: 9, 2: 9, 4: 10, 5: 6, 6: 6, 7: 11}
    assert ev.offense == [2, 6, 7] and ev.defense == [1, 4, 5]
