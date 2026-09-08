"""Calibrate the input gate that decides whether a photo is worth analysing.

The three skin models have no "none of the above" class, so argmax always names
something. A photo of a wall used to come back as a confident skin type, acne
type and severity band. This script sets the two thresholds the gate uses.

Two signals, because neither alone is enough:

  face      YuNet finds a face. Conclusive when it fires, but it only fires on
            4.3% of the acne close-ups, which are perfectly good photos of a
            cheek. Cannot be required.
  skin      Fraction of the frame passing the YCrCb + CIELAB skin test already
            used by the fairness review (scripts/estimate_skin_tone.py). Works
            on close-ups, where the face detector cannot help.

A photo passes if either fires. We want the skin threshold low enough to keep
almost every real photo and high enough to reject non-skin.

Run:
    .venv/bin/python scripts/calibrate_face_gate.py
"""

import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

from estimate_skin_tone import skin_mask

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DETECTOR = REPO_ROOT / "web" / "public" / "models" / "face_detector.onnx"

SAMPLE = 300
FACE_SCORE = 0.6


def skin_fraction(bgr: np.ndarray) -> float:
    return float(skin_mask(bgr).mean())


def synthetic_negatives() -> list[tuple[str, np.ndarray]]:
    """Things that are definitely not skin, to check the gate actually rejects."""
    rng = np.random.default_rng(0)
    out = [
        ("black", np.zeros((480, 480, 3), np.uint8)),
        ("white", np.full((480, 480, 3), 255, np.uint8)),
        ("grey wall", np.full((480, 480, 3), 128, np.uint8)),
        ("blue sky", np.dstack([
            np.full((480, 480), 220, np.uint8),
            np.full((480, 480), 150, np.uint8),
            np.full((480, 480), 80, np.uint8)]).astype(np.uint8)),
        ("green grass", np.dstack([
            np.full((480, 480), 60, np.uint8),
            np.full((480, 480), 140, np.uint8),
            np.full((480, 480), 70, np.uint8)]).astype(np.uint8)),
        ("random noise", rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)),
    ]
    # the project's own result charts: white backgrounds, coloured plot lines,
    # a realistic "user uploaded a screenshot by mistake" case
    for p in sorted(glob.glob(str(RESULTS / "*.png")))[:8]:
        # grad_cam_* and *sample_detections* are painted over real face photos,
        # so they contain genuine skin and are not negatives. A gate that fires
        # on them is behaving correctly. Excluding them keeps this honest.
        if "grad_cam" in Path(p).name or "sample_detections" in Path(p).name:
            continue
        img = cv2.imread(p)
        if img is not None:
            out.append((f"chart {Path(p).name}", img))
    return out


def main() -> None:
    random.seed(0)
    detector = cv2.FaceDetectorYN.create(str(DETECTOR), "", (640, 640), FACE_SCORE, 0.3, 5000)

    report: dict = {"sample_size": SAMPLE, "face_score_threshold": FACE_SCORE}

    for name, pattern in [("skin_type", "data/raw/skin_type/*/*/*.jpg"),
                          ("acne_type", "data/raw/acne_type/train/*/*.jpg")]:
        files = glob.glob(str(REPO_ROOT / pattern))
        random.shuffle(files)
        fracs, face_flags = [], []
        for f in files[:SAMPLE]:
            img = cv2.imread(f)
            if img is None:
                continue
            fracs.append(skin_fraction(img))
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, found = detector.detect(img)
            face_flags.append(found is not None and len(found) > 0)
        arr = np.array(fracs)
        flags = np.array(face_flags)
        faces = int(flags.sum())
        # the gate is an OR: a face, or enough skin in frame
        combined = {f"{t:.2f}": round(float((flags | (arr >= t)).mean()), 3)
                    for t in (0.05, 0.10, 0.15, 0.20)}
        report[name] = {
            "n": len(arr),
            "face_rate": round(faces / len(arr), 3),
            "skin_fraction": {
                "p1": round(float(np.percentile(arr, 1)), 3),
                "p5": round(float(np.percentile(arr, 5)), 3),
                "median": round(float(np.median(arr)), 3),
            },
            "combined_pass_rate_by_threshold": combined,
        }
        print(f"   combined gate pass rate by skin threshold: {combined}")
        print(f"{name:<12} n={len(arr):<5} face {faces / len(arr):>6.1%}   "
              f"skin p1={report[name]['skin_fraction']['p1']:.3f} "
              f"p5={report[name]['skin_fraction']['p5']:.3f} "
              f"median={report[name]['skin_fraction']['median']:.3f}")

    print("\nnegatives (should all be well under the threshold):")
    negatives = []
    for label, img in synthetic_negatives():
        frac = skin_fraction(img)
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, found = detector.detect(img)
        has_face = found is not None and len(found) > 0
        negatives.append({"label": label, "skin_fraction": round(frac, 4), "face": bool(has_face)})
        print(f"  {label:<34} skin={frac:.4f}  face={has_face}")
    report["negatives"] = negatives

    worst_negative = max(n["skin_fraction"] for n in negatives)
    floor = min(report[d]["skin_fraction"]["p5"] for d in ("skin_type", "acne_type"))
    print(f"\nhighest negative skin fraction: {worst_negative:.3f}")
    print(f"lowest real-photo 5th percentile: {floor:.3f}")
    report["highest_negative"] = round(worst_negative, 4)
    report["lowest_real_p5"] = round(floor, 4)

    out = RESULTS / "face_gate_calibration.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
