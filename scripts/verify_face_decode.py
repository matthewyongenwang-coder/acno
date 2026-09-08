"""Check the browser's hand-written YuNet decode against OpenCV's own.

web/lib/face.ts decodes YuNet's raw ONNX outputs itself, because the browser has
no OpenCV. That decode assumes a particular anchor layout: that `bbox_8` and
friends are [num_anchors, 4] laid out row-major over an (H/stride, W/stride)
grid. If that assumption is wrong (say the export is channel-first), every box
would be mislocated while still occasionally scoring above threshold, and the
face gate would look like it works while being nonsense.

The face-gate calibration numbers were measured with cv2.FaceDetectorYN, i.e.
OpenCV's C++ decoder, not the TypeScript one that actually ships. This script
closes that gap: it reimplements the TypeScript decode in Python, line for line,
runs it on the raw ONNX outputs, and compares against cv2.FaceDetectorYN on the
same images.

Run:
    .venv/bin/python scripts/verify_face_decode.py
"""

import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTOR = REPO_ROOT / "web" / "public" / "models" / "face_detector.onnx"

FACE_SIZE = 640
STRIDES = (8, 16, 32)
SCORE_THRESHOLD = 0.6
SAMPLE = 60
# OpenCV runs the net at the image's own size while the exported graph is fixed
# at 640x640, so boxes will not match to the pixel. This tolerance is a fraction
# of the image's larger side.
CENTRE_TOLERANCE = 0.06


def letterbox(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Exactly what faceTensor() does in web/lib/face.ts."""
    h, w = bgr.shape[:2]
    scale = min(FACE_SIZE / w, FACE_SIZE / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (FACE_SIZE - new_w) // 2, (FACE_SIZE - new_h) // 2
    canvas = np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(bgr, (new_w, new_h))
    return canvas, scale, pad_x, pad_y


def decode_like_typescript(outputs: dict, scale: float, pad_x: int, pad_y: int):
    """A line-for-line port of decodeFace() in web/lib/face.ts."""
    best_score, best_box = 0.0, None
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"].reshape(-1)
        obj = outputs[f"obj_{stride}"].reshape(-1)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
        cols = FACE_SIZE // stride
        for i in range(cls.shape[0]):
            c = min(max(float(cls[i]), 0.0), 1.0)
            o = min(max(float(obj[i]), 0.0), 1.0)
            score = float(np.sqrt(c * o))
            if score <= best_score:
                continue
            col, row = i % cols, i // cols
            cx = (col + bbox[i][0]) * stride
            cy = (row + bbox[i][1]) * stride
            bw = np.exp(bbox[i][2]) * stride
            bh = np.exp(bbox[i][3]) * stride
            best_score = score
            best_box = ((cx - bw / 2 - pad_x) / scale, (cy - bh / 2 - pad_y) / scale,
                        bw / scale, bh / scale)
    return best_score, best_box


def main() -> None:
    random.seed(0)
    session = ort.InferenceSession(str(DETECTOR), providers=["CPUExecutionProvider"])
    opencv = cv2.FaceDetectorYN.create(str(DETECTOR), "", (640, 640), SCORE_THRESHOLD, 0.3, 5000)
    input_name = session.get_inputs()[0].name

    files: list[str] = []
    for pattern in ("data/raw/skin_type/*/*/*.jpg", "data/raw/acne_type/train/*/*.jpg"):
        found = glob.glob(str(REPO_ROOT / pattern))
        random.shuffle(found)
        files += found[:SAMPLE // 2]

    agree = disagree = both_none = 0
    centre_errors = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]

        canvas, scale, pad_x, pad_y = letterbox(img)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]  # BGR, NCHW, 0-255
        outs = session.run(None, {input_name: blob})
        named = {o.name: v for o, v in zip(session.get_outputs(), outs)}
        ours_score, ours_box = decode_like_typescript(named, scale, pad_x, pad_y)
        ours_found = ours_score >= SCORE_THRESHOLD

        opencv.setInputSize((w, h))
        _, faces = opencv.detect(img)
        cv_found = faces is not None and len(faces) > 0

        if ours_found != cv_found:
            disagree += 1
            print(f"  DISAGREE ours={ours_found} (score {ours_score:.2f}) "
                  f"opencv={cv_found}  {Path(path).name[:44]}")
            continue
        if not ours_found:
            both_none += 1
            continue

        agree += 1
        # Compare like with like. The TypeScript decode returns the
        # highest-SCORING face, so pick OpenCV's highest-scoring one too. Taking
        # OpenCV's LARGEST face instead compares two different faces whenever an
        # image contains several, which looks like a decode error and is not.
        cv_box = max(faces, key=lambda f: f[-1])[:4]
        ours_centre = (ours_box[0] + ours_box[2] / 2, ours_box[1] + ours_box[3] / 2)
        cv_centre = (cv_box[0] + cv_box[2] / 2, cv_box[1] + cv_box[3] / 2)
        err = np.hypot(ours_centre[0] - cv_centre[0], ours_centre[1] - cv_centre[1]) / max(w, h)
        centre_errors.append(err)
        if err > CENTRE_TOLERANCE:
            print(f"  BOX OFF by {err:.1%} of the frame: {Path(path).name[:44]}")

    total = agree + disagree + both_none
    print(f"\nchecked {total} images")
    print(f"  same verdict, face found:     {agree}")
    print(f"  same verdict, no face:        {both_none}")
    print(f"  disagreed:                    {disagree}")
    if centre_errors:
        arr = np.array(centre_errors)
        print(f"  box centre error: median {np.median(arr):.2%} of frame, "
              f"max {arr.max():.2%}")
    ok = disagree == 0 and (not centre_errors or np.array(centre_errors).max() <= CENTRE_TOLERANCE)
    print("\nPASS: the shipped decode matches OpenCV" if ok else "\nFAIL: decode disagrees with OpenCV")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
