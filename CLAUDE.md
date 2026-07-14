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

Phases 0 through 3 are built and the models are trained (Colab run done, weights in
models/, gitignored). The app is a Next.js site in web/ running all three models in
the browser via onnxruntime-web; the ONNX files are committed at web/public/models/
(converted and parity-checked by scripts/convert_models.py). The earlier Streamlit
app was removed in favor of this.

Test metrics (held-out test sets): acne_type 59.3% accuracy (5 classes), lesion
detector mAP50 0.666, skin_type 42.5% (noisy dataset, overfits; honest limitation).

Deck note (results section): the saved YOLO graphs in runs/detect/results/acne_yolo/
(results.png, BoxPR_curve.png, confusion_matrix*.png) are from a throwaway 2-epoch
smoke-test run and read mAP ~0.002. They contradict the real 0.666 and must NOT go on
a slide. Only results/yolo_sample_detections.png reflects the real trained model. See
docs/presentation/DECK.md slide 5 for the caveat. Slide text for results (skin type,
acne, lesion detector) was drafted in chat this session; not stored as files.

The app also has an AI guide: web/app/api/advice/route.ts sends the scan results
(never the photo) to Claude (claude-opus-4-8), which writes a personalized analysis
and OTC product suggestions. Requires the ANTHROPIC_API_KEY environment variable;
without it the section hides itself and the rules-based routine still shows.

Still open:
1. Connect the repo to Vercel (Root Directory: web), add ANTHROPIC_API_KEY, publish.
2. Presentation redesign: follow docs/presentation/DECK.md and APPLY.md.
3. Fairness testing on diverse skin tones (Phase 4 in docs/PLAN.md).
4. Optional: improve the skin_type model (stronger regularization or backbone).

## Hard rules

- Never commit images: no dataset files, no user photos, no test selfies.
- Never commit secrets (.env, kaggle.json, API keys).
- The demo must not persist uploaded photos.
- Model output is educational guidance, never a medical diagnosis. Every report
  view includes a disclaimer and a see-a-dermatologist recommendation for
  severe findings.
- Writing style for all UI and docs: no emojis, no decorative symbols, no em dashes.

## Layout

- scripts/ - one-off utilities (data download, verification, ONNX conversion)
- src/ - Python inference pipeline (mirrors the browser app, used for verification)
- notebooks/ - the training notebook
- web/ - the Next.js app (browser-side inference, deploys on Vercel)
- docs/ - plan, data card, presentation redesign
- data/, models/, results/ - gitignored artifacts (except the two advice CSVs in data/)

## Conventions

- Commit directly to main while the team is small (same as the Moneyball Inspirit repo).
- Python 3.10+, dependencies pinned in requirements.txt.
