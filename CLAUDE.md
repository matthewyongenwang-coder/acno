# Acno - Claude Code context

Acne-analysis app for teens by a group of 5 HS students (Inspirit AI project).
Users scan their face; the app reports skin type, acne types, severity, and a
skincare guide. Motto: "Scan smarter. Skin clearer."

## Start here

1. docs/PLAN.md - architecture, phase checklist, and what to build next
2. docs/DATASETS.md - the three datasets, their classes, counts, licenses, risks
3. data/ is gitignored. To get data: `pip install -r requirements.txt` then
   `python scripts/download_data.py` then `python scripts/verify_data.py`

## Current status (update this section as work lands)

Phases 0 through 3 are built: data pipeline, the full training notebook
(notebooks/Acno.ipynb, one notebook for all three models, smoke-tested locally),
the inference pipeline (src/), and the Streamlit demo app (app/app.py).

Still open:
1. Real training run on Colab GPU (the notebook is ready, someone presses Run all),
   then put the downloaded weights in models/.
2. Deploy the app (Streamlit Community Cloud or Hugging Face Spaces).
3. Presentation redesign: follow docs/presentation/DECK.md and APPLY.md.
4. Fairness testing on diverse skin tones (Phase 4 in docs/PLAN.md).

## Hard rules

- Never commit images: no dataset files, no user photos, no test selfies.
- Never commit secrets (.env, kaggle.json, API keys).
- The demo must not persist uploaded photos.
- Model output is educational guidance, never a medical diagnosis. Every report
  view includes a disclaimer and a see-a-dermatologist recommendation for
  severe findings.
- Writing style for all UI and docs: no emojis, no decorative symbols, no em dashes.

## Layout

- scripts/ - one-off utilities (data download, verification)
- src/ - reusable pipeline code (Phase 2+)
- notebooks/ - training/exploration notebooks (Phase 1+)
- docs/ - plan, data card
- data/, models/ - gitignored artifacts

## Conventions

- Commit directly to main while the team is small (same as the Moneyball Inspirit repo).
- Python 3.10+, dependencies pinned in requirements.txt.
