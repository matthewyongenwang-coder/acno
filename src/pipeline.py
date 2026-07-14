"""Acno inference pipeline: photo in, skin report out.

Runs the three trained models on one photo and builds the full report the app
shows: skin type, acne types, lesion count, severity, an annotated image, and
a personalized routine from data/guide_rules.csv.

Usage (after downloading trained weights into models/):
    python -m src.pipeline path/to/photo.jpg

The photo is processed in memory and never written to disk. That is a hard
project rule (see docs/PLAN.md).
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RULES_CSV = REPO_ROOT / "data" / "guide_rules.csv"
ACNE_INFO_CSV = REPO_ROOT / "data" / "acne_type_info.csv"

IMG_SIZE = (224, 224)

DISCLAIMER = (
    "Acno gives educational guidance, not a medical diagnosis. "
    "For severe or persistent acne, please see a dermatologist."
)

_models = {}


def _load_models():
    """Load all three models once and cache them."""
    if _models:
        return _models
    import tensorflow as tf
    from ultralytics import YOLO

    _models["skin"] = tf.keras.models.load_model(MODELS_DIR / "skin_type.keras")
    _models["acne"] = tf.keras.models.load_model(MODELS_DIR / "acne_type.keras")
    _models["yolo"] = YOLO(str(MODELS_DIR / "acne_yolo.pt"))
    # class orders match tf.keras.utils.image_dataset_from_directory (alphabetical)
    _models["skin_classes"] = ["dry", "normal", "oily"]
    _models["acne_classes"] = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]
    return _models


def find_face(image_bgr):
    """Find the largest face and return a slightly padded crop.

    Returns (crop, found). If no face is detected we analyze the whole image
    and let the report say so.
    """
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(80, 80))
    if len(faces) == 0:
        return image_bgr, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
    x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
    x1 = min(image_bgr.shape[1], x + w + pad_w)
    y1 = min(image_bgr.shape[0], y + h + pad_h)
    return image_bgr[y0:y1, x0:x1], True


def _classify(model, class_names, face_bgr):
    """Run one classifier and return (top_class, confidence, all_scores)."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE).astype(np.float32)[None, ...]
    scores = model.predict(resized, verbose=0)[0]
    top = int(np.argmax(scores))
    all_scores = {name: round(float(s), 3) for name, s in zip(class_names, scores)}
    return class_names[top], float(scores[top]), all_scores


def _load_rules():
    rules = {}
    with open(RULES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rules[(row["skin_type"], row["severity"])] = {
                "morning": row["morning"],
                "evening": row["evening"],
                "avoid": row["avoid"],
            }
    return rules


def _load_acne_info():
    info = {}
    if ACNE_INFO_CSV.exists():
        with open(ACNE_INFO_CSV, newline="") as f:
            for row in csv.DictReader(f):
                info[row["acne_type"]] = {
                    "plain_name": row["plain_name"],
                    "explanation": row["explanation"],
                    "care_tip": row["care_tip"],
                }
    return info


def analyze(image_bgr):
    """Full analysis of one photo (BGR numpy array, as cv2 loads them).

    Returns (report_dict, annotated_rgb_image).
    """
    from src.severity import needs_dermatologist, severity_from_count

    models = _load_models()
    face, face_found = find_face(image_bgr)

    skin_type, skin_conf, skin_scores = _classify(
        models["skin"], models["skin_classes"], face)
    acne_type, acne_conf, acne_scores = _classify(
        models["acne"], models["acne_classes"], face)

    pred = models["yolo"].predict(face, conf=0.25, verbose=False)[0]
    lesion_count = len(pred.boxes)
    severity = severity_from_count(lesion_count)
    annotated_rgb = pred.plot()[:, :, ::-1]  # ultralytics plots in BGR

    rules = _load_rules()
    routine = rules.get((skin_type, severity))
    acne_info = _load_acne_info().get(acne_type)

    report = {
        "face_found": face_found,
        "skin_type": skin_type,
        "skin_type_confidence": round(skin_conf, 3),
        "skin_type_scores": skin_scores,
        "acne_type": acne_type,
        "acne_type_confidence": round(acne_conf, 3),
        "acne_type_scores": acne_scores,
        "acne_type_info": acne_info,
        "lesion_count": lesion_count,
        "severity": severity,
        "see_dermatologist": needs_dermatologist(severity, [acne_type]),
        "routine": routine,
        "disclaimer": DISCLAIMER,
    }
    return report, annotated_rgb


def analyze_path(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    return analyze(image)


def main():
    if len(sys.argv) != 2:
        print("usage: python -m src.pipeline path/to/photo.jpg")
        return 1
    report, _ = analyze_path(sys.argv[1])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
