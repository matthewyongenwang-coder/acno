"""Shared data loading for the Acno classifiers.

Two jobs:

1. Cache every image once as a uint8 array on disk. The raw datasets are 640x640
   JPEGs, and decoding them every epoch is the slowest part of training on a laptop
   CPU. Decoding once and reusing the array makes each epoch several times faster.

2. Build a leak-free test split. scripts/verify_duplicates.py found that a large
   share of the official test images also appear in the training data, which makes
   the official test accuracy look better than the model really is. We keep the
   official split so old numbers stay comparable, and add a "clean" split with every
   leaked image removed. The clean number is the one we report as real.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "raw"
CACHE = REPO_ROOT / "data" / "cache"
RESULTS = REPO_ROOT / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# images are cached slightly larger than the model input so training can take
# random crops out of them, which is a free source of augmentation
CACHE_SIZE = 256


def iter_images(split_dir: Path):
    """Yield (class_name, path) in the same order Keras would, so labels line up."""
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_dir.name, path


def class_names(dataset: str) -> list[str]:
    train_dir = DATA / dataset / "train"
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def build_cache(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (images uint8 NHWC, labels int32, relative paths) for one split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    img_file = CACHE / f"{dataset}_{split}_x.npy"
    lbl_file = CACHE / f"{dataset}_{split}_y.npy"
    path_file = CACHE / f"{dataset}_{split}_paths.json"

    if img_file.exists() and lbl_file.exists() and path_file.exists():
        return (np.load(img_file), np.load(lbl_file),
                json.loads(path_file.read_text()))

    names = class_names(dataset)
    lookup = {name: i for i, name in enumerate(names)}
    images, labels, paths = [], [], []

    for class_name, path in iter_images(DATA / dataset / split):
        with Image.open(path) as img:
            arr = np.asarray(
                img.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR),
                dtype=np.uint8,
            )
        images.append(arr)
        labels.append(lookup[class_name])
        paths.append(str(path.relative_to(REPO_ROOT)))

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int32)
    np.save(img_file, x)
    np.save(lbl_file, y)
    path_file.write_text(json.dumps(paths))
    print(f"[cache] {dataset}/{split}: {x.shape} -> {img_file.name}")
    return x, y, paths


def build_cache_at(dataset: str, split: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache a split at an arbitrary resolution, decoded from the originals.

    Needed to test whether skin texture is being lost at 224px. Upsampling the 256px
    cache would prove nothing, since that adds no detail the model did not already have.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    img_file = CACHE / f"{dataset}_{split}_x{size}.npy"
    lbl_file = CACHE / f"{dataset}_{split}_y{size}.npy"
    if img_file.exists() and lbl_file.exists():
        return np.load(img_file), np.load(lbl_file)

    names = class_names(dataset)
    lookup = {n: i for i, n in enumerate(names)}
    images, labels = [], []
    for class_name, path in iter_images(DATA / dataset / split):
        with Image.open(path) as img:
            images.append(np.asarray(
                img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8))
        labels.append(lookup[class_name])

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int32)
    np.save(img_file, x)
    np.save(lbl_file, y)
    print(f"[cache] {dataset}/{split} @ {size}px: {x.shape}")
    return x, y


def dedup_train_indices(dataset: str, paths: list[str]) -> np.ndarray:
    """Indices of training images to keep after removing redundancy and label noise.

    Two things get dropped:
      - all but one image from each group of near-identical training images, so the
        model is not rewarded for memorising whichever photo appears most often
      - every image in a group whose copies carry *different* labels, because at least
        one of those labels is wrong and we cannot tell which

    Uses the same 16x16 signature as audit_data.py.
    """
    import hashlib
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for i, rel in enumerate(paths):
        path = REPO_ROOT / rel
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L").resize((16, 16), Image.BILINEAR),
                             dtype=np.float32)
        sig = hashlib.md5((arr > arr.mean()).astype(np.uint8).tobytes()).hexdigest()
        groups[sig].append(i)
        labels_by_group[sig].add(Path(rel).parent.name)

    keep = []
    dropped_dup = dropped_conflict = 0
    for sig, members in groups.items():
        if len(labels_by_group[sig]) > 1:
            dropped_conflict += len(members)
            continue
        keep.append(members[0])
        dropped_dup += len(members) - 1

    print(f"[dedup] kept {len(keep)} of {len(paths)} training images "
          f"({dropped_dup} duplicates, {dropped_conflict} with contradictory labels)")
    return np.array(sorted(keep))


def leaked_test_paths(dataset: str) -> set[str]:
    """Test images that also appear in train or valid, per verify_duplicates.py."""
    report = RESULTS / "duplicate_pairs.json"
    if not report.exists():
        raise FileNotFoundError(
            "results/duplicate_pairs.json missing. Run scripts/verify_duplicates.py first."
        )
    return set(json.loads(report.read_text())[dataset]["test_leaked"])


def load_face_cache(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the face-cropped cache written by scripts/build_face_crops.py."""
    img_file = CACHE / f"{dataset}_face_{split}_x.npy"
    if not img_file.exists():
        raise FileNotFoundError(
            f"{img_file.name} missing. Run scripts/build_face_crops.py --dataset {dataset}"
        )
    return (
        np.load(img_file),
        np.load(CACHE / f"{dataset}_face_{split}_y.npy"),
        json.loads((CACHE / f"{dataset}_face_{split}_paths.json").read_text()),
    )


def load_splits(dataset: str, variant: str = "whole") -> dict:
    """Load train/valid/test plus the leak-free clean test split.

    variant "whole" uses the raw dataset images; "face" uses the face crops that
    match what the app actually classifies at inference time.
    """
    loader = build_cache if variant == "whole" else load_face_cache
    out = {}
    for split in ["train", "valid", "test"]:
        x, y, paths = loader(dataset, split)
        out[split] = {"x": x, "y": y, "paths": paths}

    leaked = leaked_test_paths(dataset)
    keep = np.array([p not in leaked for p in out["test"]["paths"]])
    out["clean_test"] = {
        "x": out["test"]["x"][keep],
        "y": out["test"]["y"][keep],
        "paths": [p for p, k in zip(out["test"]["paths"], keep) if k],
    }
    out["classes"] = class_names(dataset)
    out["n_leaked_removed"] = int((~keep).sum())
    out["variant"] = variant

    # optional stricter split: nearest-neighbour leak scan in feature space, which
    # also catches rotated and re-cropped copies that pixel hashing misses
    strict_file = RESULTS / "leak_scan.json"
    if strict_file.exists():
        scan = json.loads(strict_file.read_text()).get(dataset)
        if scan:
            strict = set(scan["strict_test_paths"])
            keep_s = np.array([p in strict for p in out["test"]["paths"]])
            if keep_s.any():
                out["strict_test"] = {
                    "x": out["test"]["x"][keep_s],
                    "y": out["test"]["y"][keep_s],
                    "paths": [p for p, k in zip(out["test"]["paths"], keep_s) if k],
                }
    return out


def summarise(dataset: str, splits: dict) -> None:
    names = splits["classes"]
    print(f"\n{dataset}: {len(names)} classes {names}")
    for split in ["train", "valid", "test", "clean_test"]:
        y = splits[split]["y"]
        counts = np.bincount(y, minlength=len(names))
        majority = counts.max() / len(y)
        print(f"  {split:<11} n={len(y):<5} majority-baseline={majority:.1%}  "
              f"per-class={dict(zip(names, counts.tolist()))}")
    print(f"  removed {splits['n_leaked_removed']} leaked images from the clean test split")


if __name__ == "__main__":
    for name in ["skin_type", "acne_type"]:
        summarise(name, load_splits(name))
