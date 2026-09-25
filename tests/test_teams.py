import cv2
import numpy as np

from hoopstats.teams import color_feature, fit_team_model, vote


def person(bib_bgr, bg_bgr=(200, 80, 30), rng=None):
    """Synthetic 60x120 crop: court-blue background with a coloured torso block plus noise."""
    img = np.full((120, 60, 3), bg_bgr, np.uint8)
    img[15:60, 15:45] = bib_bgr
    if rng is not None:
        img = np.clip(img.astype(int) + rng.integers(-15, 15, img.shape), 0, 255).astype(np.uint8)
    return img


def test_two_teams_separate_and_court_is_ignored():
    rng = np.random.default_rng(0)
    green, purple = (60, 200, 60), (160, 50, 140)
    court_hue = float(cv2.cvtColor(np.uint8([[[200, 80, 30]]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    feats = [color_feature(person(c, rng=rng), court_hue) for c in [green] * 20 + [purple] * 20]
    model = fit_team_model([f for f, _ in feats])
    preds = [model.predict(f, s) for f, s in feats]
    assert len(set(preds[:20])) == 1 and len(set(preds[20:])) == 1 and preds[0] != preds[20]
    # a referee in black on the blue court: only court pixels are saturated -> masked -> unknown
    f, s = color_feature(person((20, 20, 20), rng=rng), court_hue)
    assert model.predict(f, s) == "?"


def test_vote():
    assert vote(["A", "A", "B", "?"]) == "A"
    assert vote(["A", "B"]) == "?"
    assert vote(["?", "?"]) == "?"
