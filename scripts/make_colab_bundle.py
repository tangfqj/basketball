"""Pack everything the Colab pipeline notebook needs into one zip (code, calibrations, labels, models).

  uv run python scripts/make_colab_bundle.py      # -> data/colab/hoopstats_bundle.zip (~40 MB)

Uses the committed state (git archive HEAD) plus the model weights, which are not in git.
Upload the zip to Google Drive: MyDrive/hoopstats/hoopstats_bundle.zip
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

WEIGHTS = ["data/models/ball_v1.pt", "data/models/yolo11s.pt"]


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
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, commit {commit})")


if __name__ == "__main__":
    main()
