"""What counts as skin, and whether a photo is worth analysing at all.

This is the single definition of the skin test. The fairness review
(scripts/estimate_skin_tone.py) uses it to decide which pixels to measure a
tone from, and the input gate uses it to decide whether a photo contains skin
at all. web/lib/face.ts reimplements the same thresholds for the browser, and
scripts/calibrate_face_gate.py checks the two agree.
"""

import cv2
import numpy as np

# Calibrated in scripts/calibrate_face_gate.py against 300 images per dataset
# plus non-skin negatives. See results/face_gate_calibration.json.
SKIN_FRACTION_THRESHOLD = 0.10
FACE_SCORE_THRESHOLD = 0.6


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of likely skin pixels.

    YCrCb thresholds separate skin from background better than RGB. We additionally
    require a positive CIELAB b*, because skin is always yellowish: pink backgrounds,
    grey clothing and red annotation arrows are not, and letting them through was what
    put pale close-ups into the "dark" bin on the first attempt.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_star = lab[..., 1].astype(np.int16) - 128
    b_star = lab[..., 2].astype(np.int16) - 128

    return (
        (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 127)
        & (y >= 30) & (y <= 240)          # drop crushed blacks and blown highlights
        & (b_star >= 6)                   # skin is yellowish; pink/grey things are not
        & (a_star >= 3) & (a_star <= 45)   # plausible red component for skin
    )


def skin_fraction(bgr: np.ndarray) -> float:
    """Fraction of the frame that looks like skin."""
    return float(skin_mask(bgr).mean())
