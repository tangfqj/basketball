"""Runs only when the TrackID3x3 ground truth is present under data/ (git-ignored)."""

from pathlib import Path

import pytest

from hoopstats.court import get_court, is_on_court
from hoopstats.datasets import trackid3x3 as t

GT = Path(__file__).resolve().parents[1] / "data" / "trackid3x3_repo" / "ground_truth"
pytestmark = pytest.mark.skipif(not GT.exists(), reason="TrackID3x3 ground truth not downloaded")


def test_mot_and_delimitation():
    dets = t.load_mot(GT / "Outdoor" / "MOT" / "IMG_0104.txt")
    assert len(dets) > 10_000
    flow = t.load_delimitation(GT / "Outdoor" / "delimitation_frames" / "IMG_0104.csv")
    assert flow[0].label == "Game start"
    teams = t.team_membership(flow)
    assert sorted(teams) == sorted({d.track_id for d in dets})


def test_outdoor_calibration_maps_players_onto_court():
    cal = t.outdoor_calibration(GT)
    court = get_court(cal.court_standard)
    dets = [d for d in t.load_mot(GT / "Outdoor" / "MOT" / "IMG_0104.txt") if d.frame == 200]
    pts = cal.image_to_court([d.foot_point for d in dets])
    assert all(is_on_court(court, x, y, margin=1.0) for x, y in pts)
