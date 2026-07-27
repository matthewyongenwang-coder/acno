#!/usr/bin/env bash
# Round 2. Round 1 found two things: face crops beat whole images, and lighter
# augmentation beats heavier augmentation by a wide margin. This round separates
# those two effects and then tries stronger backbones on the winning setup.
#
# Usage: bash scripts/sweep_round2.sh
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")

run() {
  echo ""
  echo "########## $* ##########"
  $PY train_classifier.py "$@" 2>&1 \
    | grep -vE "WARNING|warnings|oneDNN|cpu_feature|TF-TRT|self\."
}

# Was the round-1 gain from the face crop, or purely from lighter augmentation?
run --dataset skin_type --variant whole --backbone mobilenetv2 --tag whole_mnv2_light --augment light

# If less augmentation keeps helping, does none at all help more?
run --dataset skin_type --variant face --backbone mobilenetv2 --tag face_mnv2_none --augment none

# Lock in the acne_type winner with its weights kept this time.
run --dataset acne_type --variant whole --backbone mobilenetv2 --tag acne_mnv2_med --augment medium

# Stronger backbones on the best skin_type setup so far.
run --dataset skin_type --variant face --backbone efficientnetv2b1 --tag face_effb1_light --augment light
run --dataset skin_type --variant face --backbone resnet50v2 --tag face_r50_light --augment light

echo ""
echo "########## leaderboard ##########"
$PY leaderboard.py
