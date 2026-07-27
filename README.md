# Acno

Scan smarter. Skin clearer.

Acno helps teens going through puberty understand and take care of their skin. Scan your
face and get an instant analysis: your skin type, what kind of acne you have, how severe
it is, and a simple guide for keeping your skin clean. Built by a team of 5 high school
students as an Inspirit AI project.

Acno gives educational guidance, not medical diagnosis. For severe or persistent acne,
see a dermatologist.

## How it works

1. Skin type model: classifies dry / normal / oily skin
2. Acne type model: classifies whiteheads, blackheads, papules, pustules, cysts
3. Lesion detector: finds and counts acne lesions to estimate severity
4. Guide engine: turns the results into a personalized skincare routine

## Getting started

```bash
git clone https://github.com/matthewyongenwang-coder/acno.git
cd acno
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py   # fetches the 3 public Kaggle datasets (no account needed)
python scripts/verify_data.py     # confirms everything downloaded correctly
```

Datasets are never stored in this repo (privacy and licensing: they contain photos of
real people's faces). See docs/DATASETS.md for sources, labels, and known risks.

## Training the models

Everything trains in one notebook: [notebooks/Acno.ipynb](notebooks/Acno.ipynb).
Open it in Google Colab, switch the runtime to a GPU, and Run all. It downloads the
data itself, trains all three models, saves every chart, and ends with a download of
the trained weights. Unzip those into the repo root so `models/` has
`skin_type.keras`, `acne_type.keras`, and `acne_yolo.pt`.

For the local, scriptable version of the same recipe, plus the hyperparameter search
and the experiment ledger, see [docs/TRAINING.md](docs/TRAINING.md).

## How well do the models actually work

- [docs/RESULTS.md](docs/RESULTS.md) - current accuracy for all three models, every
  configuration tried, and per-class recall. Generated from the experiment ledger.
- [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md) - **read this before quoting any
  number.** Both classifier datasets ship with images shared between their training and
  test splits, which made the accuracy we originally published too high.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) - what the models are for, what they are not
  for, and where they fail.

## Running the app

The app is a Next.js site in `web/` that runs all three models directly in the
browser with onnxruntime-web. There is no backend: the photo is analyzed on the
user's own device and never uploaded anywhere.

```bash
cd web
npm install
npm run dev
```

The ONNX models are committed at `web/public/models/` (converted from the trained
weights, see below), so the app works straight from a fresh clone.

### The AI guide (optional but worth it)

Alongside the three vision models, the app can ask an AI language model to write
a personalized analysis and suggest over-the-counter products based on the scan
results. Privacy is preserved: only the scan numbers are sent, never the photo.
Without an API key the app simply skips this section and falls back to its
built-in advice.

To enable it, set one of these environment variables (whichever key you have):
`ANTHROPIC_API_KEY` for Claude, or `OPENAI_API_KEY` for OpenAI. Locally, put it
in `web/.env.local`; on Vercel, add it under Project Settings, Environment
Variables.

### Deploying on Vercel

Import the GitHub repo in Vercel and set the project's Root Directory to `web`.
Add the `ANTHROPIC_API_KEY` environment variable if you want the AI guide.
Everything else is default: Vercel detects Next.js, builds, and serves the models
as static files.

### Updating the models

After a new training run, convert the fresh weights (this also verifies the ONNX
outputs match the originals) and commit the updated files:

```bash
pip install tf2onnx onnx onnxruntime
python scripts/convert_models.py
```

Every report carries the disclaimer: educational guidance, not medical diagnosis.

## Project docs

- [docs/PLAN.md](docs/PLAN.md): architecture, roadmap, and security rules
- [docs/DATASETS.md](docs/DATASETS.md): data card for all three datasets
- [CITATIONS.md](CITATIONS.md): every source we used (datasets, research, libraries, AI, advice)
- [docs/presentation/DECK.md](docs/presentation/DECK.md): slide-by-slide presentation plan
- [CLAUDE.md](CLAUDE.md): context for AI-assisted development sessions

## Team

Travis, Matthew, Alan, Tanner, Erwin

## License

Acno's code and docs are MIT licensed: see [LICENSE](LICENSE). Use it, change it,
build on it, just keep the copyright notice.

Two files are not ours to relicense. The lesion detector weights were trained with
Ultralytics YOLOv8, which is AGPL-3.0, and the training images belong to their dataset
authors. [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) spells out exactly which
files that covers and what it means if you reuse them.
