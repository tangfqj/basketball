import random

from hoopstats.training.ball_dataset import _crop_origin


def test_crop_contains_box_with_margin():
    rng = random.Random(0)
    for box in ([100, 100, 30, 30], [3700, 2100, 34, 34], [1900, 1000, 40, 40]):
        for _ in range(50):
            ox, oy = _crop_origin(3840, 2160, 1280, 1280, box, rng)
            assert 0 <= ox <= 3840 - 1280 and 0 <= oy <= 2160 - 1280
            assert ox <= box[0] and box[0] + box[2] <= ox + 1280
            assert oy <= box[1] and box[1] + box[3] <= oy + 1280


def test_negative_crop_inside_image():
    rng = random.Random(1)
    for _ in range(50):
        ox, oy = _crop_origin(1920, 1080, 1280, 1080, None, rng, fallback=(1800, 50))
        assert 0 <= ox <= 640 and oy == 0
