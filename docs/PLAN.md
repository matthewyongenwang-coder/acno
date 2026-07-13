# Acno Development Plan

App: teens scan their face, get an analysis of skin type and acne, plus a personalized
guide for keeping their skin clean. Team of 5: Travis, Matthew, Alan, Tanner, Erwin.
Motto: "Scan smarter. Skin clearer."

Presentation section owners (from the planning doc "Inspirit AI - Acno"):
Background + Why AI (Travis), Dataset Exploration (Travis and Matthew), Models Used
(Alan and Tanner), Results + Graphs (Alan), Demo (Matthew), Future Applications (Tanner),
Conclusions (Erwin).

## Architecture overview

Three models feed one report:

1. Skin type classifier: dry / normal / oily (transfer learning, MobileNetV2 or
   EfficientNet-B0 backbone) on `data/raw/skin_type`.
2. Acne type classifier: whiteheads / blackheads / papules / pustules / cyst on
   `data/raw/acne_type`.
3. Lesion detector: YOLOv8-nano on `data/raw/acne_yolo`; lesion count maps to a
   severity level (clear / mild / moderate / severe).

App flow: photo in -> face detected and cropped -> three models run -> report page with
skin type, acne types found, severity, annotated image, and a routine guide generated
from a rules table (skin type x severity -> recommendations). Rules live in a data file,
not hardcoded, so non-coders on the team can edit advice text.

## Phases

### Phase 0: Repo and data (DONE)

- [x] Repo structure, .gitignore, requirements.txt
- [x] Dataset selection and download pipeline (scripts/download_data.py)
- [x] Data verification (scripts/verify_data.py)
- [x] Data card with licenses and risks (docs/DATASETS.md)

### Phase 1: Baseline models (NEXT)

- [ ] Notebook or script: fine-tune MobileNetV2 on skin_type, report accuracy + confusion matrix
- [ ] Same for acne_type (watch the Whiteheads class imbalance, use class weights)
- [ ] Train YOLOv8n on acne_yolo (ultralytics package), report mAP and sample detections
- [ ] Save all metrics/plots to a results/ folder for Alan's Results + Graphs section
- Target: something that runs end to end, even if accuracy is mediocre

### Phase 2: Inference pipeline

- [ ] `src/pipeline.py`: image path in, JSON report out (all three models)
- [ ] Face detection/crop preprocessing step (mediapipe or opencv haar cascade)
- [ ] Severity mapping from lesion count (document thresholds in docs/)
- [ ] Recommendation rules table (skin type x severity -> guide text)

### Phase 3: Demo app (Matthew owns the demo)

- [ ] Streamlit app (same stack the team knows from the Moneyball project): upload or
      webcam photo -> report page
- [ ] Annotated image with detected lesions (YOLO boxes)
- [ ] Plain-English explanations next to every term
- [ ] Disclaimer on every report: not medical advice, see a dermatologist for severe acne

### Phase 4: Evaluation and fairness

- [ ] Confusion matrices and per-class precision/recall for both classifiers
- [ ] Test on diverse skin tones; document where the model fails
- [ ] Grad-CAM visualizations for explainability slides

## Security and privacy rules (non-negotiable)

1. No dataset images, user photos, or team selfies ever get committed. The .gitignore
   enforces this for data/, test_photos/, selfies/.
2. No secrets in the repo. No API keys, no kaggle.json, no .env files.
3. The demo must not store uploaded photos. Process in memory, show the report, done.
   If we ever add saving, it needs explicit consent language first.
4. Faces of minors are sensitive data. We only train on public research datasets, never
   on photos of classmates.
5. All advice output is educational guidance, not medical diagnosis. Severe/cystic
   findings should always recommend seeing a dermatologist.

## Conventions

- Commit directly to main while the repo is small (same convention as the Moneyball
  Inspirit project). Switch to PRs if two people start editing the same code.
- Python 3.10+; pin dependencies in requirements.txt.
- Keep notebooks in notebooks/, reusable code in src/, one-off scripts in scripts/.
