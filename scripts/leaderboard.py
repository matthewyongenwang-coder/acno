"""Print every training run so far, best first.

train_classifier.py appends one JSON line per run to results/experiments.jsonl.
This reads that ledger so the search stays honest: every configuration we tried is
listed, not just the one that happened to win.

Run:
    .venv/bin/python scripts/leaderboard.py [--dataset skin_type]
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "results" / "experiments.jsonl"


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--sort", default="clean_test", choices=["clean_test", "test", "valid"])
    args = p.parse_args()

    rows = load()
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    if not rows:
        print("no runs recorded yet")
        return

    def best_clean(r: dict) -> float:
        """Best clean-test accuracy this run can deliver, using TTA where measured.

        Flip test-time augmentation can be baked into the exported ONNX graph, so it
        costs the browser one extra forward pass and nothing else. Where a run
        measured it, that is the number we could actually ship.
        """
        plain = r["metrics"]["clean_test"]["accuracy"]
        tta = r.get("metrics_tta", {}).get("clean_test", {}).get("accuracy")
        return max(plain, tta) if tta is not None else plain

    rows.sort(key=lambda r: (best_clean(r) if args.sort == "clean_test"
                             else r["metrics"][args.sort]["accuracy"]), reverse=True)

    header = (f"{'tag':<28}{'dataset':<10}{'backbone':<20}{'aug':<8}"
              f"{'valid':>7}{'test':>7}{'clean':>7}{'+TTA':>7}{'F1':>7}{'min':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r["metrics"]
        tta = r.get("metrics_tta", {}).get("clean_test", {})
        tta_acc = f"{tta['accuracy']:>7.3f}" if tta else f"{'-':>7}"
        f1 = max(m["clean_test"]["macro_f1"], tta.get("macro_f1", 0)) if tta \
            else m["clean_test"]["macro_f1"]
        print(f"{r['tag'][:27]:<28}{r['dataset']:<10}{r['backbone'][:19]:<20}"
              f"{r['augment']:<8}"
              f"{m['valid']['accuracy']:>7.3f}{m['test']['accuracy']:>7.3f}"
              f"{m['clean_test']['accuracy']:>7.3f}{tta_acc}{f1:>7.3f}"
              f"{r['train_seconds'] / 60:>6.1f}")

    print(f"\n{len(rows)} runs. 'clean' is accuracy on the leak-free test split, "
          f"which is the number we report. '+TTA' adds mirror-image averaging.")


if __name__ == "__main__":
    main()
