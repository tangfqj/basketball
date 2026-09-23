import pytest

from hoopstats.rules import points_for
from hoopstats.schema import Zone


@pytest.mark.parametrize("rules,zone,expected", [
    ("5v5", Zone.INSIDE_ARC, 2), ("5v5", Zone.BEYOND_ARC, 3), ("5v5", Zone.FT, 1),
    ("3x3", Zone.INSIDE_ARC, 1), ("3x3", Zone.BEYOND_ARC, 2), ("3x3", Zone.FT, 1),
])
def test_points(rules, zone, expected):
    assert points_for(rules, zone, made=True) == expected
    assert points_for(rules, zone, made=False) == 0


def test_unknown_rules():
    with pytest.raises(ValueError):
        points_for("4v4", Zone.FT, True)
