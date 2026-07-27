"""Generate docs/RESULTS.md from the experiment ledger.

The results tables in our documentation are produced from results/experiments.jsonl
rather than typed by hand, so they cannot drift from what actually ran. Re-run this
after any training run:

    .venv/bin/python scripts/write_report.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"

# what we published before this work, for the before/after comparison
PREVIOUS = {
    "skin_type": (0.425, "official test split, which is 11% contaminated"),
    "acne_type": (0.593, "official test split, which is 49% contaminated"),
}
TARGET = 0.60


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(x) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def headline_split(run: dict) -> str:
    """Strictest split this run was measured on."""
    return "strict_test" if "strict_test" in run["metrics"] else "clean_test"


def best_accuracy(run: dict, split: str) -> tuple[float, bool]:
    """Best accuracy on `split`, and whether test-time augmentation produced it."""
    plain = run["metrics"][split]["accuracy"]
    tta = run.get("metrics_tta", {}).get(split, {}).get("accuracy")
    if tta is not None and tta > plain:
        return tta, True
    return plain, False


def shipped_tag(dataset: str) -> str | None:
    """Which run's weights are actually in web/public/models/."""
    meta = REPO_ROOT / "web" / "public" / "models" / "models.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get(dataset, {}).get("tag")


def pick_best(runs: list[dict], dataset: str) -> dict | None:
    """The run we ship, not the run that happened to score highest.

    Picking the best-scoring run and quoting its number is cherry-picking: with a
    114-image evaluation set, the highest score in a sweep is partly luck. We report
    the model that is actually deployed, and quote run-to-run variance separately.
    """
    pool = [r for r in runs
            if r["dataset"] == dataset
            and not r["tag"].startswith(("smoke", "benchmark", "debug"))]
    if not pool:
        return None
    tag = shipped_tag(dataset)
    for r in reversed(pool):
        if r["tag"] == tag:
            return r
    return max(pool, key=lambda r: best_accuracy(r, headline_split(r))[0])


def strict_eval(dataset: str, tag: str) -> dict | None:
    """Strict-split scores measured after the fact by evaluate_torch.py.

    Runs that finished before the feature-space leak scan existed never saw the strict
    split during training, so their ledger entry lacks it.
    """
    path = RESULTS / f"eval_torch_{dataset}_{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("strict_test")


def seed_spread(runs: list[dict], dataset: str) -> list[dict]:
    """Runs that differ only by random seed, for a variance estimate."""
    return [r for r in runs
            if r["dataset"] == dataset and r["tag"].startswith("seed")]


def section_headline(runs: list[dict], yolo: list[dict]) -> list[str]:
    lines = [
        "## Headline results",
        "",
        "Every figure below is measured on the **strictest split available**, not the",
        "official one. Both classifier datasets share images between their own train and",
        "test splits, so the official numbers, including the ones we published earlier,",
        "are inflated. See [DATA_QUALITY.md](DATA_QUALITY.md).",
        "",
        "- **clean split**: official test, minus every image confirmed by pixel comparison",
        "  to also appear in train or valid.",
        "- **strict split**: also minus every image whose nearest training neighbour in",
        "  feature space is near-identical. This catches the rotated and re-cropped copies",
        "  that pixel comparison misses, which is the dominant problem in acne_type.",
        "",
        "| Model | Published before | Now | 95% CI | Majority baseline | n | 60% target |",
        "|---|---|---|---|---|---|---|",
    ]

    for dataset in ["skin_type", "acne_type"]:
        best = pick_best(runs, dataset)
        if not best:
            continue

        after = strict_eval(dataset, best["tag"])
        if after:
            acc = max(after["plain"]["accuracy"], after["tta"]["accuracy"])
            used_tta = after["tta"]["accuracy"] >= after["plain"]["accuracy"]
            n, base = after["n"], after["baseline"]
            split_name = "strict"
        else:
            split = headline_split(best)
            acc, used_tta = best_accuracy(best, split)
            cm = np.array(best["metrics"][split]["confusion_matrix"])
            n = int(cm.sum())
            base = float(cm.sum(axis=1).max() / n) if n else 0.0
            split_name = "strict" if split == "strict_test" else "clean"

        # Where we have repeated seeds, the headline is their average, not the best
        # one. Quoting the top run of a sweep on a 114-image split reports luck.
        seeds = seed_spread(runs, dataset)
        if seeds:
            accs = [acc] + [
                (strict_eval(dataset, s["tag"]) or {}).get("plain", {}).get(
                    "accuracy", s["metrics"][headline_split(s)]["accuracy"])
                for s in seeds]
            acc = sum(accs) / len(accs)
            suffix = f" (mean of {len(accs)} seeds, range {pct(min(accs))}-{pct(max(accs))})"
        else:
            suffix = " (with TTA)" if used_tta else ""

        lo, hi = wilson(int(round(acc * n)), n)
        prev_acc, prev_note = PREVIOUS[dataset]
        verdict = "**met**" if lo >= TARGET else ("borderline" if hi >= TARGET else "not met")
        lines.append(
            f"| {dataset} | {pct(prev_acc)} ({prev_note}) | **{pct(acc)}**{suffix} "
            f"({split_name}) | [{pct(lo)}, {pct(hi)}] | {pct(base)} | {n} | {verdict} |")

    if yolo:
        kept = next((r for r in yolo if r["tag"] == "v1_original"), None)
        retrained = next((r for r in yolo if r["tag"] != "v1_original"), None)
        if kept:
            m = kept["test"]
            verdict = "**met**" if m["mAP50"] >= TARGET else "not met"
            lines.append(
                f"| lesion detector | 66.6% mAP50 | **{pct(m['mAP50'])} mAP50** "
                f"(test split) | n/a | n/a | 48 | {verdict} |")

    lines += ["", "### About the detector", ""]
    if yolo and retrained:
        lines += [
            "We retrained the detector for 120 epochs and it came out **worse**: "
            f"{pct(retrained['test']['mAP50'])} mAP50 against "
            f"{pct(kept['test']['mAP50'])} for the weights already shipping, measured on "
            "the same held-out test split. So we kept the original weights and changed "
            "nothing. The retrain is left in the ledger because a negative result is "
            "still a result.",
            "",
            "The old figure was also being quoted from the validation split, which is "
            "what early stopping optimises against. Both numbers above are now measured "
            "on the test split, which no part of training saw.",
            "",
        ]
    return lines


def section_variance(runs: list[dict]) -> list[str]:
    seeds = seed_spread(runs, "skin_type")
    shipped = pick_best(runs, "skin_type")
    if len(seeds) < 2 or not shipped:
        return []

    rows = []
    for r in [shipped] + seeds:
        m = r["metrics"]
        split = "strict_test" if "strict_test" in m else "clean_test"
        after = strict_eval("skin_type", r["tag"])
        acc = after["plain"]["accuracy"] if after else m[split]["accuracy"]
        rows.append((r["tag"], r.get("seed", "?"), m["valid"]["accuracy"], acc))

    accs = [a for _, _, _, a in rows]
    lines = [
        "## Why we do not quote skin_type's best run",
        "",
        "The same configuration, retrained with nothing changed but the random seed:",
        "",
        "| run | seed | validation | strict test |",
        "|---|---|---|---|",
    ]
    for tag, seed, val, acc in rows:
        lines.append(f"| {tag} | {seed} | {pct(val)} | {pct(acc)} |")
    lines += [
        "",
        f"Identical settings produce results spanning {pct(min(accs))} to {pct(max(accs))}, "
        f"a spread of {pct(max(accs) - min(accs))}. The evaluation set is small enough "
        "that this is mostly luck.",
        "",
        "Worse, **validation accuracy does not predict test accuracy here**: the run with "
        "the best validation score has the worst test score. That means we cannot use "
        "validation to pick the good run, and a sweep's top score is not a real finding.",
        "",
        f"So the honest summary for skin_type is roughly "
        f"**{pct(sum(accs) / len(accs))} give or take several points**, against a 37.7% "
        "baseline. It is a genuine improvement on the 42.5% we published before (which "
        "was below its own baseline), but it is not 60% and it will not become 60% by "
        "tuning. The limit is the dataset. See [DATA_QUALITY.md](DATA_QUALITY.md).",
        "",
    ]
    return lines


def section_runs(runs: list[dict]) -> list[str]:
    lines = ["## Every configuration we tried", "",
             "Including the ones that made things worse. The search history is the evidence",
             "that the final numbers are not one lucky run.", ""]
    for dataset in ["skin_type", "acne_type"]:
        pool = [r for r in runs if r["dataset"] == dataset]
        if not pool:
            continue
        pool.sort(key=lambda r: best_accuracy(r, headline_split(r))[0], reverse=True)
        lines += [f"### {dataset}", "",
                  "| run | framework | input | backbone | augmentation | valid | official "
                  "test | clean | strict | macro F1 | minutes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in pool:
            m = r["metrics"]
            strict = pct(m["strict_test"]["accuracy"]) if "strict_test" in m else "-"
            lines.append(
                f"| {r['tag']} | {r.get('framework', 'keras')} | {r.get('variant', 'whole')} "
                f"| {r['backbone']} | {r['augment']} | {pct(m['valid']['accuracy'])} | "
                f"{pct(m['test']['accuracy'])} | {pct(m['clean_test']['accuracy'])} | "
                f"{strict} | {m['clean_test']['macro_f1']:.3f} | "
                f"{r['train_seconds'] / 60:.1f} |")
        lines.append("")
    return lines


def section_per_class(runs: list[dict]) -> list[str]:
    lines = ["## Per-class recall of the chosen models", "",
             "Overall accuracy can hide a class the model never gets right. Whiteheads is "
             "about a quarter the size of the other acne classes, so it is the one to "
             "watch.", ""]
    for dataset in ["skin_type", "acne_type"]:
        best = pick_best(runs, dataset)
        if not best:
            continue
        split = headline_split(best)
        source = best.get("metrics_tta", best["metrics"])
        recalls = source.get(split, best["metrics"][split])["per_class_recall"]
        lines += [f"### {dataset} ({best['tag']}, {split})", "",
                  "| class | recall |", "|---|---|"]
        for cls, rec in recalls.items():
            lines.append(f"| {cls} | {pct(rec)} |")
        lines.append("")
    return lines


def section_reading(runs: list[dict]) -> list[str]:
    skin = pick_best(runs, "skin_type")
    lines = ["## How to read these numbers", ""]
    if skin:
        after = strict_eval("skin_type", skin["tag"])
        if after:
            n, acc = after["n"], after["plain"]["accuracy"]
        else:
            split = headline_split(skin)
            cm = np.array(skin["metrics"][split]["confusion_matrix"])
            n = int(cm.sum())
            acc, _ = best_accuracy(skin, split)
        lo, hi = wilson(int(round(acc * n)), n)
        lines += [
            f"**skin_type's evaluation set is only {n} images.** Its confidence interval "
            f"runs from {pct(lo)} to {pct(hi)}, which is wide enough to change what "
            "conclusion you draw. Treat any single figure for it as approximate, and do "
            "not report a change of a few points as an improvement.",
            "",
        ]
    lines += [
        "**Always quote the baseline.** Guessing the most common class every time scores "
        "about 38% on skin_type and about 27% on acne_type. An accuracy figure without "
        "that context does not mean anything.",
        "",
        "**Macro F1 matters more than accuracy for acne_type**, because the classes are "
        "unbalanced and Whiteheads is small.",
        "",
    ]
    return lines


def main() -> None:
    runs = load(RESULTS / "experiments.jsonl")
    yolo = load(RESULTS / "yolo_experiments.jsonl")
    if not runs:
        raise SystemExit("no runs in results/experiments.jsonl yet")

    lines = ["# Results", "",
             f"Generated by `scripts/write_report.py` on {date.today().isoformat()}. "
             "Do not edit by hand; re-run the script instead.", ""]
    lines += section_headline(runs, yolo)
    lines += section_variance(runs)
    lines += section_reading(runs)
    lines += section_per_class(runs)
    lines += section_runs(runs)

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(runs)} classifier runs, {len(yolo)} detector runs)")


if __name__ == "__main__":
    main()
