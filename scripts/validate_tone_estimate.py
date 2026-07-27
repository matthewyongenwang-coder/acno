"""Check whether the estimated skin tone is measuring skin tone, or something else.

Before any fairness number is reported, the tone estimate has to survive two checks:

1. **Label independence.** If estimated tone correlates strongly with the class label,
   the estimate is probably picking up the condition rather than the person. This is a
   real risk on acne_type, whose images are close-ups where inflamed skin is red: a
   redness-driven estimate would tag "cyst" images as darker, and any apparent fairness
   gap would just be the label in disguise.

2. **Face availability.** ITA is defined for skin. On images that are mostly not skin,
   or where no face is present to anchor what skin looks like, the estimate is a guess.

What this script prints decides how much the fairness review is allowed to claim.

Run:
    .venv/bin/python scripts/validate_tone_estimate.py
"""

import json
from pathlib import Path

import numpy as np

import acno_data

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def summarise(dataset: str, report: dict) -> dict:
    print(f"\n{'=' * 78}\n{dataset}\n{'=' * 78}")
    split = report[dataset]["train"]
    ita = np.array([np.nan if v is None else v for v in split["ita"]], dtype=float)
    labels = np.array(split["label"])
    classes = acno_data.class_names(dataset)
    ok = ~np.isnan(ita)

    print(f"usable estimate on {ok.sum()} of {len(ita)} training images "
          f"({ok.mean():.1%})")
    print(f"ITA spread: p5 {np.nanpercentile(ita, 5):.0f}, "
          f"median {np.nanmedian(ita):.0f}, p95 {np.nanpercentile(ita, 95):.0f}")

    print("\nmean estimated ITA per class (lower = darker):")
    per_class = {}
    for c, name in enumerate(classes):
        sel = ok & (labels == c)
        if sel.sum() == 0:
            continue
        per_class[name] = float(np.mean(ita[sel]))
        print(f"    {name:<14} n={sel.sum():>5}  mean ITA {per_class[name]:>6.1f}")

    spread = max(per_class.values()) - min(per_class.values()) if per_class else 0.0
    total_spread = float(np.nanpercentile(ita, 95) - np.nanpercentile(ita, 5))
    ratio = spread / total_spread if total_spread else 0.0

    print(f"\n    spread across classes: {spread:.1f} degrees")
    print(f"    spread across images:  {total_spread:.1f} degrees (p5 to p95)")
    print(f"    ratio: {ratio:.1%}")

    if ratio > 0.25:
        verdict = ("FAIL: estimated tone tracks the label too closely. Any fairness gap "
                   "measured with it would partly be the label in disguise.")
    elif ratio > 0.12:
        verdict = ("WEAK: some association with the label. Report gaps only as "
                   "suggestive, and never as a headline number.")
    else:
        verdict = "OK: tone estimate is largely independent of the label."
    print(f"\n    {verdict}")

    return {"usable_fraction": float(ok.mean()), "per_class_mean_ita": per_class,
            "class_spread": spread, "image_spread": total_spread,
            "ratio": ratio, "verdict": verdict}


def main() -> None:
    report = json.loads((RESULTS / "skin_tone.json").read_text())
    out = {name: summarise(name, report) for name in ["skin_type", "acne_type"]}
    (RESULTS / "tone_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'tone_validation.json'}")


if __name__ == "__main__":
    main()
