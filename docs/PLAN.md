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

### Phase 1: Baseline models (notebook DONE, real training run pending)

All three models live in one notebook, notebooks/Acno.ipynb, written to run top to
bottom on a Colab GPU. It has been smoke-tested end to end locally in quick-test mode.
What remains is the real training run (Runtime > Run all on a T4, roughly 30 to 45
minutes), then dropping the downloaded weights into models/.

- [x] Notebook: fine-tune MobileNetV2 on skin_type, report accuracy + confusion matrix
- [x] Same for acne_type (Whiteheads class imbalance handled with class weights)
- [x] Train YOLOv8n on acne_yolo (ultralytics package), report mAP and sample detections
- [x] Save all metrics/plots to a results/ folder for Alan's Results + Graphs section
      (results/ is gitignored because sample charts contain dataset faces)
- [ ] Run the real (non quick-test) training on Colab GPU and bring weights into models/

### Phase 2: Inference pipeline (DONE)

- [x] `src/pipeline.py`: image path in, JSON report out (all three models)
- [x] Face detection/crop preprocessing step (opencv haar cascade; opencv pinned
      below 5 because OpenCV 5 removed that API)
- [x] Severity mapping from lesion count (src/severity.py, thresholds documented there
      and in the notebook)
- [x] Recommendation rules table (data/guide_rules.csv and data/acne_type_info.csv,
      plain text files committed to git so non-coders can edit the advice)

### Phase 3: Web app (DONE, Vercel hookup pending)

The app is a Next.js site in web/ that runs all three models in the browser via
onnxruntime-web (ONNX files committed at web/public/models/, converted by
scripts/convert_models.py). No backend at all: the photo never leaves the device,
which is our privacy rule made literal. An earlier Streamlit version was replaced
by this so the project deploys on Vercel like the team's other projects.

- [x] Upload or camera photo -> report page
- [x] Annotated image with detected lesions (YOLO boxes drawn on canvas)
- [x] Plain-English explanations next to every term
- [x] Disclaimer on every report: not medical advice, see a dermatologist for severe acne
- [x] Browser inference verified to match the Python pipeline (same photo: 33 lesions both ways)
- [ ] Connect the repo to Vercel (Root Directory setting: web)

Run it locally: `cd web && npm install && npm run dev`.

### Phase 4: Evaluation and fairness

- [x] Confusion matrices and per-class precision/recall for both classifiers (in the notebook)
- [ ] Test on diverse skin tones; document where the model fails
- [x] Grad-CAM visualizations for explainability slides (in the notebook)

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
