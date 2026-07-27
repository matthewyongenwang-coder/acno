"""Estimate a skin tone group for every image, since no dataset records one.

None of the three Kaggle datasets document skin tone coverage. That is the single
biggest known gap in the project, and it cannot be measured without a tone label per
image, so we estimate one.

Method: Individual Typology Angle (ITA), the standard proxy used in dermatology
imaging. For skin pixels, convert to CIELAB and compute

    ITA = arctan((L* - 50) / b*) * 180 / pi

Higher ITA means lighter skin. The cut points below are the ones commonly used in the
dermatology literature to map ITA onto Fitzpatrick-like bands.

Picking skin pixels matters more than the formula. We:
  1. build a skin mask in YCrCb, which separates skin from background better than RGB
  2. drop the darkest and brightest 20% of those pixels, which removes shadows, hair
     and specular highlights (a shiny forehead would otherwise read as very light skin)
  3. take the median of what is left, so a few stray pixels cannot move the answer

Honest limitation, stated up front and repeated in docs/FAIRNESS.md: these are
uncontrolled photographs from the web, not calibrated dermatology images. Lighting,
white balance, filters and makeup all shift ITA. This estimate is good enough to ask
"does the model work worse on darker skin", and not good enough to assign anyone a
Fitzpatrick type. We use it for the former only.

Run:
    .venv/bin/python scripts/estimate_skin_tone.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import acno_data

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
SCRATCH = Path("/private/tmp/claude-501/-Users-matthewwang/"
               "f2ca490c-47b6-49d2-8c8d-84f756964e11/scratchpad")

# ITA cut points, lightest first. Standard dermatology bands, collapsed to four groups
# because finer bins would leave too few images per group to say anything.
TONE_BANDS = [
    ("very light", 55.0, 200.0),
    ("light", 28.0, 55.0),
    ("intermediate", -30.0, 28.0),
    ("dark", -200.0, -30.0),
]
MIN_SKIN_PIXELS = 200


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


def image_ita(bgr: np.ndarray) -> float | None:
    """Median ITA over mid-luminance skin pixels, or None if we cannot tell.

    Uses the standard ITA definition, arctan((L* - 50) / b*), which assumes b* > 0.
    Returning None is deliberate: an image where we cannot find enough confident skin
    should be excluded from the fairness numbers rather than guessed at.
    """
    mask = skin_mask(bgr)
    if mask.sum() < MIN_SKIN_PIXELS:
        return None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV packs L into 0-255 and a/b into 0-255 with a 128 offset
    L = lab[..., 0] * (100.0 / 255.0)
    b = lab[..., 2] - 128.0

    Ls, bs = L[mask], b[mask]
    # drop shadows and specular highlights before averaging
    lo, hi = np.percentile(Ls, [20, 80])
    keep = (Ls >= lo) & (Ls <= hi)
    if keep.sum() < MIN_SKIN_PIXELS // 4:
        keep = np.ones_like(Ls, dtype=bool)

    L_med = float(np.median(Ls[keep]))
    b_med = float(np.median(bs[keep]))
    if b_med <= 1.0:
        return None
    return float(np.degrees(np.arctan((L_med - 50.0) / b_med)))


def band_for(ita: float | None) -> str:
    if ita is None:
        return "unknown"
    for name, lo, hi in TONE_BANDS:
        if lo <= ita < hi:
            return name
    return "unknown"


def analyse(dataset: str, use_face_crop: bool = False) -> dict:
    """Estimate tone per image.

    use_face_crop matters enormously for skin_type. Those images are marketing photos
    with coloured backgrounds, and a pink or orange backdrop passes a YCrCb skin test
    happily. Measured on the whole frame, a Black woman on a pink background landed in
    the *lightest* quartile and light-skinned people on warm backgrounds landed in the
    darkest. Restricting to the detected face region fixes that, at the cost of only
    covering the images where a face was actually found.
    """
    print(f"\n{'=' * 78}\n{dataset}"
          f"{'  (face region only)' if use_face_crop else ''}\n{'=' * 78}")
    out = {}
    for split in ["train", "valid", "test"]:
        base_x, y, paths = acno_data.build_cache(dataset, split)
        if use_face_crop:
            face_x, _, face_paths = acno_data.load_face_cache(dataset, split)
            found = np.load(acno_data.CACHE / f"{dataset}_face_{split}_found.npy")
            by_path = {p: (face_x[i], bool(found[i])) for i, p in enumerate(face_paths)}
        itas, bands = [], []
        for i, image in enumerate(base_x):
            if use_face_crop:
                crop, has_face = by_path.get(paths[i], (None, False))
                if not has_face:
                    itas.append(None)
                    bands.append("unknown")
                    continue
                image = crop
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ita = image_ita(bgr)
            itas.append(ita)
            bands.append(band_for(ita))
        out[split] = {"ita": itas, "band": bands, "paths": paths,
                      "label": y.tolist(), "face_region": use_face_crop}

        counts = {name: bands.count(name) for name, _, _ in TONE_BANDS}
        counts["unknown"] = bands.count("unknown")
        total = len(bands)
        print(f"\n[{split}] {total} images")
        for name, n in counts.items():
            bar = "#" * int(40 * n / max(1, total))
            print(f"    {name:<14}{n:>5}  {n / total:>6.1%}  {bar}")
    return out


def contact_sheets(dataset: str, report: dict, use_face_crop: bool = False) -> None:
    """Save a montage per tone band so the binning can be checked by eye."""
    if use_face_crop:
        x, _, _ = acno_data.load_face_cache(dataset, "train")
    else:
        x, _, _ = acno_data.build_cache(dataset, "train")
    bands = report["train"]["band"]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, _, _ in TONE_BANDS:
        idx = [i for i, b in enumerate(bands) if b == name][:8]
        if not idx:
            continue
        tiles = [x[i] for i in idx]
        while len(tiles) < 8:
            tiles.append(np.zeros_like(x[0]))
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        sheet = np.concatenate(rows, axis=0)
        path = SCRATCH / f"{dataset}_tone_bands.png"
        Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(path)
        print(f"\n    contact sheet (rows lightest to darkest) -> {path}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {}
    # skin_type images are composed marketing photos, so the background has to be
    # excluded; acne_type images are close-ups that are almost entirely skin already
    for dataset, use_face in [("skin_type", True), ("acne_type", False)]:
        report[dataset] = analyse(dataset, use_face_crop=use_face)
        contact_sheets(dataset, report[dataset], use_face_crop=use_face)

    out = RESULTS / "skin_tone.json"
    out.write_text(json.dumps(report))
    print(f"\nwrote {out}")
    print("\nCheck the contact sheets before trusting any number that follows.")


if __name__ == "__main__":
    main()
