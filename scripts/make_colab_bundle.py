"""Pack everything the Colab notebooks need into one zip (code, calibrations, labels, models, dataset GT).

  uv run python scripts/make_colab_bundle.py      # -> data/colab/hoopstats_bundle.zip (~40 MB)

Uses the committed state (git archive HEAD) plus the model weights, which are not in git.
Upload the zip to Google Drive: MyDrive/hoopstats/hoopstats_bundle.zip
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

WEIGHTS = ["data/models/ball_v1.pt", "data/models/yolo11s.pt", "data/models/bib_v1.pt"]
# TrackID3x3 ground truth used on Colab (bib reader crops): dev-video player boxes + all game-flow files
GT = Path("data/trackid3x3_repo/ground_truth/Outdoor")
GT_MOT_VIDEOS = ["IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107", "IMG_0108", "IMG_0109", "IMG_0110"]   # dev split


def main():
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        print("warning: uncommitted changes are NOT included (the bundle uses the last commit):\n" + dirty)
    out = Path("data/colab/hoopstats_bundle.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=zip", "--prefix=hoopstats/", "-o", str(out), "HEAD"], check=True)
    missing = [w for w in WEIGHTS if not Path(w).exists()]
    if missing:
        sys.exit(f"missing model weights: {missing}")
    with zipfile.ZipFile(out, "a") as z:
        for w in WEIGHTS:
            z.write(w, f"hoopstats/{w}")
        gt_files = [GT / "MOT" / f"{v}.txt" for v in GT_MOT_VIDEOS] + sorted((GT / "delimitation_frames").glob("*.csv"))
        for f in gt_files:
            if f.exists():
                z.write(f, f"hoopstats/{f}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, commit {commit})")


if __name__ == "__main__":
    main()
