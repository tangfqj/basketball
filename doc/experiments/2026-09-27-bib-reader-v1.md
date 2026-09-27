# Milestone 2, step 3 — bib reader v1

Notebook `notebooks/train_bib_reader.ipynb` (Colab A100), script `scripts/train_bib_reader.py`.
YOLO11n-cls, 128 px letterboxed torso crops (every 8th frame, dataset boxes), 20 epochs, no flips,
training on crops with ≤ 30% of the torso covered by other players. Leave-one-video-out over
IMG_0104–0108 (~8k crops per video). Results in `data/bib_results/` (not in git).

| Held-out video | Crop acc (clean) | Acc / coverage at conf ≥ 0.8 | 3-s window vote (free) | 3-s window vote (roster) |
|---|---|---|---|---|
| IMG_0104 | 98.7% | 99.2% / 71% | 99.1% | 99.5% |
| IMG_0105 | 99.5% | 99.7% / 76% | 99.4% | 100% |
| IMG_0106 | 99.6% | 99.8% / 73% | 99.5% | 99.7% |
| IMG_0107 | 98.7% | 99.4% / 73% | 99.8% | 100% |
| IMG_0108 | 99.8% | 99.9% / 75% | 99.8% | 99.8% |
| **All** | **99.3%** | 99.6% / 74% | 99.5% | **99.85%** (4007 windows) |

Every number ≥ 99.6% with the roster vote. The bib reader is **not** the bottleneck: player
accuracy will be limited by shooter attribution (89.1%, step 2).

## Caveat: it partly recognises players, not only digits

- Crop accuracy (99.3%) is far above the share of crops with a legible number (~65–78% by eye, step 0).
  Inspecting IMG_0108 crops: side views with **no visible digit** are still predicted correctly with
  confidence 1.00.
- Reason: the same groups of players wear the same bibs in several videos (e.g. green #6/#9/#11 in
  IMG_0104 team B and IMG_0108 team A; yellow #6/#9/#10 in IMG_0107 team A and IMG_0108 team B). A
  held-out *video* is therefore not held-out *players*; the model also uses appearance (build, sleeves,
  shorts, skin).
- For this dataset that is harmless and even helpful (the test videos come from the same event).
  It means the numbers above do **not** show how well it reads bibs of players it has never seen.
  A stricter check would hold out every video of one bib group (colour + numbers).
