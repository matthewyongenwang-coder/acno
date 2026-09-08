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

from src.skin import FACE_SCORE_THRESHOLD, SKIN_FRACTION_THRESHOLD, skin_fraction

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
FACE_DETECTOR = REPO_ROOT / "web" / "public" / "models" / "face_detector.onnx"
RULES_CSV = REPO_ROOT / "data" / "guide_rules.csv"
ACNE_INFO_CSV = REPO_ROOT / "data" / "acne_type_info.csv"

IMG_SIZE = (224, 224)

DISCLAIMER = (
    "Acno gives educational guidance, not a medical diagnosis. "
    "For severe or persistent acne, please see a dermatologist."
)

NOT_SKIN_WARNING = (
    "We could not find a face in this photo, and very little of the frame looks "
    "like skin, so there is nothing to report. Try a clear, well-lit photo of "
    "your face, or a close-up of the patch of skin you are asking about."
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


def _load_face_detector():
    if "face" not in _models:
        _models["face"] = cv2.FaceDetectorYN.create(
            str(FACE_DETECTOR), "", (640, 640), FACE_SCORE_THRESHOLD, 0.3, 5000)
    return _models["face"]


def find_face(image_bgr):
    """Find the largest face and return a slightly padded crop.

    Returns (crop, found). If no face is detected we analyze the whole image
    and let the report say so.

    Uses YuNet rather than the old Haar cascade, so this agrees with the browser
    (web/lib/face.ts runs the same ONNX file). Haar missed faces at any angle and
    could not be run in the browser without shipping all of OpenCV.
    """
    detector = _load_face_detector()
    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    if faces is None or len(faces) == 0:
        return image_bgr, False

    # faces rows are [x, y, w, h, 5 landmark pairs..., score]
    x, y, fw, fh = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
    pad_w, pad_h = int(fw * 0.15), int(fh * 0.15)
    x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
    x1 = min(w, x + fw + pad_w)
    y1 = min(h, y + fh + pad_h)
    if x1 <= x0 or y1 <= y0:
        return image_bgr, False
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

    # The classifiers have no "none of the above" class, so argmax names a skin
    # type even for a photo of a wall. Gate on two signals before reporting:
    # a detected face, or enough of the frame looking like skin. A face alone is
    # not enough, because it only fires on 4.3% of legitimate acne close-ups.
    skin = skin_fraction(image_bgr)
    if not face_found and skin < SKIN_FRACTION_THRESHOLD:
        report = {
            "face_found": False,
            "skin_fraction": round(skin, 4),
            "gate_passed": False,
            "warning": NOT_SKIN_WARNING,
            "disclaimer": DISCLAIMER,
        }
        return report, image_bgr[:, :, ::-1]

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
        "skin_fraction": round(skin, 4),
        "gate_passed": True,
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
