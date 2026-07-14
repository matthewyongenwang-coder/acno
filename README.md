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

### Deploying on Vercel

Import the GitHub repo in Vercel and set the project's Root Directory to `web`.
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
- [docs/presentation/DECK.md](docs/presentation/DECK.md): slide-by-slide presentation plan
- [CLAUDE.md](CLAUDE.md): context for AI-assisted development sessions

## Team

Travis, Matthew, Alan, Tanner, Erwin
