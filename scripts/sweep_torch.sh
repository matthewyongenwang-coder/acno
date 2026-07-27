#!/usr/bin/env bash
# MPS sweep. One run at a time on purpose: the GPU is the bottleneck, and running
# several at once would only add heat and memory pressure without finishing sooner.
#
# Usage: bash scripts/sweep_torch.sh [stage]
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
STAGE="${1:-1}"

run() {
  echo ""
  echo "########## $* ##########"
  # no output filter here: BSD grep has no lookahead, and a broken filter silently
  # killed a whole sweep once already
  $PY train_torch.py "$@" 2>&1
}

if [ "$STAGE" = "1" ]; then
  # skin_type overfits within a few epochs (train loss 1.07 -> 0.23 while validation
  # accuracy stays flat), which is what weak labels look like. So this stage pushes
  # regularisation hard rather than reaching for bigger backbones: EfficientNetV2B1
  # already scored worse than MobileNetV2 in the TensorFlow sweep.
  COMMON="--epochs 30 --patience 10 --dropout 0.5 --weight-decay 0.05"

  for BB in mobilenet_v3_large mobilenet_v2 resnet18; do
    run --dataset skin_type --variant face --backbone "$BB" \
        --tag "t_face_${BB}_med" --augment medium $COMMON
  done

  run --dataset skin_type --variant face --backbone mobilenet_v2 \
      --tag t_face_mobilenet_v2_strong --augment strong $COMMON

  for BB in mobilenet_v3_large mobilenet_v2; do
    run --dataset skin_type --variant whole --backbone "$BB" \
        --tag "t_whole_${BB}_med" --augment medium $COMMON
  done

  # acne_type is already at 89.6% from TensorFlow; see whether torch can beat it.
  for BB in mobilenet_v3_large efficientnet_b0; do
    run --dataset acne_type --variant whole --backbone "$BB" \
        --tag "t_acne_${BB}" --augment light --epochs 30 --patience 10
  done
fi

if [ "$STAGE" = "2" ]; then
  # Stage 2 is filled in once stage 1 names a winner.
  echo "edit stage 2 before running it"
fi

echo ""
echo "########## leaderboard ##########"
$PY leaderboard.py
