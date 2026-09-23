import numpy as np

from hoopstats.calibration import Calibration
from hoopstats.court import get_court, is_beyond_arc, is_on_court

C = get_court("fiba_3x3")


def test_zones():
    assert not is_beyond_arc(C, 0.0, C.hoop_y)                        # under the basket
    assert not is_beyond_arc(C, 0.0, C.hoop_y + C.arc_radius - 0.01)  # just inside the top of the arc
    assert is_beyond_arc(C, 0.0, C.hoop_y + C.arc_radius + 0.01)      # just outside
    assert not is_beyond_arc(C, C.corner_x - 0.01, 0.5)               # corner, inside the line
    assert is_beyond_arc(C, C.corner_x + 0.01, 0.5)                   # corner three
    assert is_beyond_arc(C, 0.0, C.hoop_y + C.arc_radius)  is False   # on the line counts as inside


def test_on_court():
    assert is_on_court(C, 0, 5)
    assert not is_on_court(C, 10, 5)


def test_homography_roundtrip(tmp_path):
    court = [(-7.5, 0), (7.5, 0), (7.5, 11), (-7.5, 11)]
    image = [(100, 400), (1800, 400), (1900, 1000), (0, 1000)]
    cal = Calibration("fiba_3x3", (1920, 1080), image, court)
    assert cal.reprojection_error() < 1e-6
    p = np.array([[950.0, 700.0]])
    assert np.allclose(cal.court_to_image(cal.image_to_court(p)), p)
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json").image_points == cal.image_points
