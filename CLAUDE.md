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

Models were retrained in July 2026. Current metrics live in docs/RESULTS.md, generated
from results/experiments.jsonl by scripts/write_report.py. Do not quote metrics from
memory or from older docs; regenerate the report instead.

IMPORTANT before quoting any accuracy: both classifier datasets share images between
their own train and test splits (about 49% of the acne_type test set, 11% of
skin_type's). The acne_type dataset was exported from Roboflow with augmentation
applied before the split. Every number we published before July 2026 was inflated by
this. We now evaluate on a leak-free "clean" split and report that. See
docs/DATA_QUALITY.md.

Always quote the majority-class baseline next to an accuracy number. skin_type's old
42.5% was below its own 44.0% baseline, meaning the model was worse than always
answering "normal". The cause was a training bug, not the data: the notebook saved the
last epoch rather than the best one. See docs/TRAINING.md.

Deck note (results section): the saved YOLO graphs in runs/detect/results/acne_yolo/
(results.png, BoxPR_curve.png, confusion_matrix*.png) are from a throwaway 2-epoch
smoke-test run and read mAP ~0.002. They contradict the real 0.666 and must NOT go on
a slide. Only results/yolo_sample_detections.png reflects the real trained model. See
docs/presentation/DECK.md slide 5 for the caveat. Slide text for results (skin type,
acne, lesion detector) was drafted in chat this session; not stored as files.

The app also has an AI guide: web/app/api/advice/route.ts sends the scan results
(never the photo) to Gemini (gemini-3.8-flash, via @google/genai's Interactions
API, with Google Search grounding for product research and thinking_level "high"
for the report), which writes a personalized analysis and OTC product suggestions.
Requires the GEMINI_API_KEY environment variable; without it the section hides
itself and the rules-based routine still shows. Switched from Claude/OpenAI in
September 2026 - self-funded, and Gemini's pricing plus native search grounding
fit the reasoning/research bar better than paying for Opus.

Known inconsistency: src/pipeline.py crops to a detected face before classifying, but
web/lib/analyze.ts (what users actually run) classifies the whole photo. The shipped
models are trained on whole images to match the browser. Training on face crops scores
several points higher, but adopting it means classifying a crop, and the shipped weights
expect whole frames.

There IS now a face detector in the browser (YuNet, web/lib/face.ts), but it is an input
GATE, not a crop: it decides whether the photo shows skin at all. Do not start cropping
to it without retraining, or every prediction silently changes.

Shipped models (July 2026), all measured on the strict leak-free split. These are a
summary only; docs/RESULTS.md is the canonical source and is regenerated from the
experiment ledger. If a number here disagrees with RESULTS.md, RESULTS.md is right.
- acne_type  98.8% with TTA (baseline 26.8%), MobileNetV3-Large, mirror TTA baked
  into the graph. TREAT THIS AS AN UPPER BOUND, NOT FIELD ACCURACY: a frozen
  ImageNet probe, fine-tuned on nothing, scores 73.5% on the same strict split
  against that 26.8% baseline, and still scores 63.9% with the middle half of
  every image blanked out, where the lesion has to be. The class is largely
  predictable from colour and background, not lesions. See docs/DATA_QUALITY.md.
  Do not "fix" this with a source-aware resplit; the signal is class-level, not
  duplicate-level, so a resplit will not move it.
- skin_type  ~44% and unstable (baseline 37.7%), MobileNetV3-Large
- acne_yolo  0.666 mAP50 on test, UNCHANGED. A 120-epoch retrain scored worse (0.641)
  so the original weights were kept.

skin_type is unstable: the same config across three seeds gave 50.9/38.6/43.9%, and
validation accuracy does not predict test accuracy. Do not quote its best run, and do
not treat a change under ten points as real.

Classifiers are now trained in PyTorch on MPS (scripts/train_torch.py), about 5x faster
than the TensorFlow path. Both paths are kept; see docs/TRAINING.md.

FAIRNESS REVIEW DONE (July 2026, docs/FAIRNESS.md). Tone is estimated per image via
Individual Typology Angle; groups are QUARTILES of the training distribution, never
Fitzpatrick types, because absolute ITA does not survive uncontrolled lighting. Two
traps found the hard way: arctan2 breaks when b* <= 0 (pink images read as dark), and
on skin_type the coloured marketing backgrounds contaminate the skin mask, so tone MUST
be measured inside the detected face region there. Always eyeball the contact sheets.
Result: acne_type shows no meaningful tone gap and the test had power to detect one
(>7 points); skin_type and the detector show none but are far too small to conclude
anything (detectable gap 54 and 38 points). Never claim Acno works equally well on dark
skin: the datasets contain almost none. `--augment tone` raised every acne_type group
and narrowed the spread 4.4 -> 2.4 points, so it ships.

Still open:
1. Presentation redesign: follow docs/presentation/DECK.md and APPLY.md.
2. A tone-diverse evaluation set is now the highest-value missing piece. A few hundred
   labelled images would take skin_type's detectable gap from 54 points to under 10.
   Natural ask for Jocelyn's clinic dermatologists.
3. skin_type is capped by its dataset, not the recipe. Bigger backbones did WORSE, and
   tone augmentation changed nothing. A better labelled dataset is the only large win.
4. Decide whether to add browser-side face detection, which would let the app use the
   stronger face-cropped models.
5. Browser download grew to 46 MB from ~30 MB. Float16 quantisation would roughly
   halve it but needs testing against onnxruntime-web's wasm backend first.

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
