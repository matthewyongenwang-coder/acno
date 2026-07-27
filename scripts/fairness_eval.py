"""Measure how the shipped models perform across skin tone groups.

This is the first milestone of the funded project plan, and the biggest known gap in
the project: none of the three datasets record skin tone, so we have never been able to
say whether Acno works as well for everyone.

Two decisions shape everything here, both forced by the data:

**Groups are quartiles, not Fitzpatrick types.** scripts/estimate_skin_tone.py shows
that Individual Typology Angle tracks apparent tone in the right *order*, but its
absolute value on uncontrolled web photos is pushed around by exposure and white
balance as much as by melanin. So an absolute claim like "5% of our data is Fitzpatrick
VI" is not defensible, while "the darkest-appearing quarter of our data" is. We split
at the quartiles of the training distribution and apply those same cut points to test.

**Everything is measured on the strict, leak-free split.** Splitting an already small
test set four ways leaves few images per group, so every number is printed with a
confidence interval and a sample size, and small differences must not be read as real.

Run:
    .venv/bin/python scripts/fairness_eval.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import acno_data
from estimate_skin_tone import image_ita
from train_torch import AcnoNet, device, predict, score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"

GROUP_NAMES = ["lightest quarter", "second", "third", "darkest quarter"]


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def quartile_edges(train_ita: list) -> list[float]:
    vals = np.array([v for v in train_ita if v is not None], dtype=float)
    return [float(np.percentile(vals, q)) for q in (25, 50, 75)]


def assign_group(ita, edges: list[float]) -> int | None:
    """0 = lightest quarter (highest ITA) ... 3 = darkest quarter."""
    if ita is None:
        return None
    if ita >= edges[2]:
        return 0
    if ita >= edges[1]:
        return 1
    if ita >= edges[0]:
        return 2
    return 3


def shipped(dataset: str) -> tuple[str, str]:
    meta = json.loads((REPO_ROOT / "web" / "public" / "models" / "models.json").read_text())
    entry = meta[dataset]
    return entry["tag"], entry.get("variant", "whole")


def evaluate_classifier(dataset: str, tone: dict) -> dict:
    tag, variant = shipped(dataset)
    ckpt = torch.load(MODELS / f"{dataset}__{tag}.pt", map_location="cpu",
                      weights_only=False)
    dev = device()
    model = AcnoNet(ckpt["backbone"], len(ckpt["classes"]), ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    splits = acno_data.load_splits(dataset, variant=variant)
    classes = splits["classes"]
    split_name = "strict_test" if "strict_test" in splits else "clean_test"
    data = splits[split_name]

    edges = quartile_edges(tone[dataset]["train"]["ita"])
    ita_by_path = dict(zip(tone[dataset]["test"]["paths"], tone[dataset]["test"]["ita"]))
    groups = np.array([
        assign_group(ita_by_path.get(p), edges) if ita_by_path.get(p) is not None else -1
        for p in data["paths"]
    ])

    probs = predict(model, data["x"], data["y"], 32, dev, tta=True)
    pred = probs.argmax(axis=1)
    y = data["y"]

    print(f"\n{'=' * 78}\n{dataset}  ({tag}, {split_name}, n={len(y)})\n{'=' * 78}")
    print(f"ITA quartile edges from training data: {[round(e, 1) for e in edges]}")
    print(f"\n{'group':<20}{'n':>5}{'accuracy':>11}{'95% CI':>20}{'macro F1':>10}")
    print("-" * 78)

    overall = float((pred == y).mean())
    rows = []
    for g, name in enumerate(GROUP_NAMES):
        sel = groups == g
        n = int(sel.sum())
        if n == 0:
            print(f"{name:<20}{0:>5}{'':>11}{'no images':>20}")
            rows.append({"group": name, "n": 0})
            continue
        acc = float((pred[sel] == y[sel]).mean())
        lo, hi = wilson(int(round(acc * n)), n)
        sub = score(probs[sel], y[sel], classes)
        rows.append({"group": name, "n": n, "accuracy": acc,
                     "ci95": [lo, hi], "macro_f1": sub["macro_f1"],
                     "per_class_recall": sub["per_class_recall"]})
        print(f"{name:<20}{n:>5}{acc:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
              f"{sub['macro_f1']:>10.3f}")

    unknown = int((groups == -1).sum())
    scored = [r for r in rows if r["n"] > 0]
    gap = (max(r["accuracy"] for r in scored) - min(r["accuracy"] for r in scored)
           if scored else 0.0)
    print(f"\noverall {overall:.3f}   spread between best and worst group {gap:.3f}"
          f"   ({unknown} images had no usable tone estimate)")

    # is the spread bigger than sampling noise? compare against overlapping intervals
    overlap = all(
        any(a["ci95"][0] <= b["ci95"][1] and b["ci95"][0] <= a["ci95"][1]
            for b in scored if b is not a)
        for a in scored) if len(scored) > 1 else True
    print("    every group's interval overlaps at least one other: "
          f"{'yes, so the spread is within noise' if overlap else 'NO, the spread looks real'}")

    return {"tag": tag, "split": split_name, "overall": overall, "edges": edges,
            "groups": rows, "gap": gap, "unknown": unknown,
            "within_noise": bool(overlap)}


def evaluate_detector(tone_edges_source: str = "acne_type") -> dict:
    """Lesion detector mAP on the lighter half versus the darker half of its test set."""
    from ultralytics import YOLO

    data_dir = acno_data.DATA / "acne_yolo" / "test" / "images"
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png"})
    itas = []
    for p in paths:
        bgr = cv2.imread(str(p))
        itas.append(image_ita(bgr) if bgr is not None else None)

    usable = [(p, v) for p, v in zip(paths, itas) if v is not None]
    if len(usable) < 8:
        print("\nlesion detector: too few usable tone estimates to split")
        return {}
    median = float(np.median([v for _, v in usable]))

    print(f"\n{'=' * 78}\nlesion detector (test split, n={len(paths)})\n{'=' * 78}")
    print(f"splitting at median ITA {median:.1f}; only two groups because 48 images "
          f"cannot support four")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(MODELS / "acne_yolo.pt"))
    out = {}
    tmp = RESULTS / "fairness_yolo"
    tmp.mkdir(parents=True, exist_ok=True)

    for label, keep in [("lighter half", lambda v: v >= median),
                        ("darker half", lambda v: v < median)]:
        subset = [p for p, v in usable if keep(v)]
        listing = tmp / f"{label.replace(' ', '_')}.txt"
        listing.write_text("\n".join(str(p.resolve()) for p in subset) + "\n")
        yaml = tmp / f"{label.replace(' ', '_')}.yaml"
        yaml.write_text(f"path: {tmp.resolve()}\ntrain: {listing.name}\n"
                        f"val: {listing.name}\nnc: 1\nnames: ['Acne']\n")
        res = model.val(data=str(yaml), imgsz=640, device=dev, verbose=False)
        out[label] = {"n": len(subset), "mAP50": float(res.box.map50),
                      "precision": float(res.box.mp), "recall": float(res.box.mr)}
        print(f"    {label:<14} n={len(subset):>3}  mAP50 {res.box.map50:.3f}  "
              f"P {res.box.mp:.3f}  R {res.box.mr:.3f}")

    if len(out) == 2:
        vals = [v["mAP50"] for v in out.values()]
        print(f"    gap in mAP50: {abs(vals[0] - vals[1]):.3f}")
    return out


def main() -> None:
    tone = json.loads((RESULTS / "skin_tone.json").read_text())
    report = {name: evaluate_classifier(name, tone) for name in ["skin_type", "acne_type"]}
    try:
        report["lesion_detector"] = evaluate_detector()
    except Exception as exc:
        print(f"\nlesion detector evaluation skipped: {exc}")

    (RESULTS / "fairness.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS / 'fairness.json'}")


if __name__ == "__main__":
    main()
