"""Build a face-cropped copy of the skin_type cache.

The app does not classify whole photos. web/lib/analyze.ts and src/pipeline.py both
detect the largest face first and classify a padded crop of it. But the models were
trained on the raw dataset images, which are scraped web photos: collages, product
shots, faces at every scale, wide backgrounds. So the model was trained on one
distribution and used on another.

This script closes that gap by applying the exact same face detection used at
inference time (OpenCV's frontal-face Haar cascade, 15% padding, largest face wins)
to every training image, and caching the crops. Images where no face is found keep a
centre crop, which is what the app falls back to as well.

Run:
    .venv/bin/python scripts/build_face_crops.py --dataset skin_type
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import acno_data

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "data" / "cache"
CACHE_SIZE = acno_data.CACHE_SIZE

_cascade = None


def cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def crop_face(path: Path) -> tuple[np.ndarray, bool]:
    """Mirror src/pipeline.py find_face(), then resize to the cache size."""
    image = cv2.imread(str(path))
    if image is None:
        with Image.open(path) as img:
            image = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(80, 80))
    found = len(faces) > 0
    if found:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
        x1 = min(image.shape[1], x + w + pad_w)
        y1 = min(image.shape[0], y + h + pad_h)
        image = image[y0:y1, x0:x1]
    else:
        # same fallback the app uses: keep the whole frame, squared off centrally
        h, w = image.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        image = image[y0:y0 + side, x0:x0 + side]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8), found


def build(dataset: str, split: str) -> tuple[int, int]:
    img_file = CACHE / f"{dataset}_face_{split}_x.npy"
    lbl_file = CACHE / f"{dataset}_face_{split}_y.npy"
    path_file = CACHE / f"{dataset}_face_{split}_paths.json"
    meta_file = CACHE / f"{dataset}_face_{split}_found.npy"

    names = acno_data.class_names(dataset)
    lookup = {n: i for i, n in enumerate(names)}
    images, labels, paths, founds = [], [], [], []

    for class_name, path in acno_data.iter_images(acno_data.DATA / dataset / split):
        arr, found = crop_face(path)
        images.append(arr)
        labels.append(lookup[class_name])
        paths.append(str(path.relative_to(REPO_ROOT)))
        founds.append(found)

    np.save(img_file, np.stack(images))
    np.save(lbl_file, np.asarray(labels, dtype=np.int32))
    np.save(meta_file, np.asarray(founds, dtype=bool))
    path_file.write_text(json.dumps(paths))
    return sum(founds), len(founds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="skin_type")
    args = p.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"building face crops for {args.dataset}")
    for split in ["train", "valid", "test"]:
        found, total = build(args.dataset, split)
        print(f"  {split:<6} face detected in {found}/{total} images ({found / total:.1%})")
    print("\nA low detection rate is itself a finding: those images are collages,")
    print("product shots or extreme close-ups rather than usable face photos.")


if __name__ == "__main__":
    main()
