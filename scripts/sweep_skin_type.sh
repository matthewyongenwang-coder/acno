#!/usr/bin/env bash
# Search for a skin_type recipe that beats the majority-class baseline by a real
# margin. Runs sequentially: the laptop has one CPU pool and parallel runs just
# make each other slower.
#
# Usage: bash scripts/sweep_skin_type.sh
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")

run() {
  echo ""
  echo "########## $* ##########"
  $PY train_classifier.py --dataset skin_type "$@" 2>&1 \
    | grep -vE "WARNING|warnings|oneDNN|cpu_feature|TF-TRT|self\."
}

# Does training on face crops (what the app actually classifies) beat training on
# the raw scraped images?
run --variant whole --backbone mobilenetv2 --tag whole_mnv2_med --augment medium
run --variant face  --backbone mobilenetv2 --tag face_mnv2_med  --augment medium

# Is the heavy photometric augmentation helping or destroying the signal?
run --variant face  --backbone mobilenetv2 --tag face_mnv2_light --augment light

echo ""
echo "########## leaderboard ##########"
$PY leaderboard.py --dataset skin_type
