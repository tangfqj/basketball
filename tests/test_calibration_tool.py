import numpy as np

from hoopstats.calibration.tool import build_calibration
from hoopstats.court import court_landmarks, court_lines, get_court


def test_build_calibration_with_rim(tmp_path):
    court = get_court("fiba_3x3")
    lm = court_landmarks(court)
    # a synthetic camera: court metres -> pixels by an affine map
    pix = {n: [960 + 60 * x, 900 - 40 * y] for n, (x, y) in lm.items()}
    pts = {n: pix[n] for n in ("baseline_left_corner", "baseline_right_corner", "ft_line_left", "ft_line_right")}
    pts.update(rim_left=[940, 830], rim_right=[980, 830], rim_front=[960, 835], rim_back=[960, 825])
    cal = build_calibration("fiba_3x3", (1920, 1080), pts, lm)
    assert cal.reprojection_error() < 1e-6
    assert cal.rim_center == (960, 830)
    assert len(cal.rim_edge) == 4
    top = cal.court_to_image(np.array([lm["arc_top"]]))[0]
    assert np.allclose(top, pix["arc_top"], atol=1e-3)
    half = cal.scaled(960, 540)
    assert half.rim_center == (480, 415)
    cal.save(tmp_path / "c.json")


def test_court_lines_shape():
    lines = court_lines(get_court("fiba"))
    assert len(lines) == 8 and all(len(ln) >= 2 for ln in lines)
