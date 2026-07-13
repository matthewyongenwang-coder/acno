"""Download all Acno training datasets from Kaggle into data/raw/.

Usage:
    pip install -r requirements.txt
    python scripts/download_data.py

No Kaggle account or API key is required: all three datasets are public and
kagglehub supports anonymous downloads. Downloads are cached in
~/.cache/kagglehub, then copied into data/raw/ so the whole project reads
from one predictable place.

Datasets (see docs/DATASETS.md for full details and licenses):
  1. skin_type   - dry / normal / oily face classification
  2. acne_type   - whiteheads / blackheads / papules / pustules / cyst
  3. acne_yolo   - acne lesion detection, YOLOv8 format (for severity counts)
"""

import shutil
import sys
from pathlib import Path

import kagglehub

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# name -> (kaggle slug, subfolder inside the download that holds the data)
DATASETS = {
    "skin_type": ("shakyadissanayake/oily-dry-and-normal-skin-types-dataset", "Oily-Dry-Skin-Types"),
    "acne_type": ("tiswan14/acne-dataset-image", "AcneDataset"),
    "acne_yolo": ("osmankagankurnaz/acne-dataset-in-yolov8-format", "data-2"),
}


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (slug, subdir) in DATASETS.items():
        dest = RAW_DIR / name
        if dest.exists():
            print(f"[skip] {name}: already exists at {dest}")
            continue
        print(f"[download] {name} <- {slug}")
        cache_path = Path(kagglehub.dataset_download(slug))
        src = cache_path / subdir
        if not src.exists():
            # Layout changed upstream: fall back to the whole download
            print(f"  warning: expected subfolder {subdir!r} not found, copying full download")
            src = cache_path
        shutil.copytree(src, dest)
        print(f"  -> {dest}")
    print("\nDone. Run scripts/verify_data.py to check everything is in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
