#!/usr/bin/env bash
# ETTm1 experiments with mixed prediction (noise head below stop_step, x_start above).
# Trains + tests every horizon using scripts/ettm1/config_<H>.yaml.
#
# Usage (from repo root):
#   bash scripts/ettm1/run.sh
#   CSV=data/ETTm1.csv GPU=1 HORIZONS="96 192" bash scripts/ettm1/run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

HORIZONS=${HORIZONS:-"96 192 336 720"}
PREDICTION_TYPE=${PREDICTION_TYPE:-mixed}
GPU=${GPU:-0}
CSV=${CSV:-}
TAG=${TAG:-${PREDICTION_TYPE}}

LOG_DIR=logs/ettm1
mkdir -p "$LOG_DIR"

for H in $HORIZONS; do
    NAME=ettm1_${TAG}_${H}
    echo "==== ${NAME} ===="
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config scripts/ettm1/config_${H}.yaml \
        ${CSV:+--csv "$CSV"} \
        --prediction_type "$PREDICTION_TYPE" \
        --use_early_stop \
        --checkpoint_dir checkpoints/${NAME} \
        2>&1 | tee "${LOG_DIR}/${NAME}.log"
done

echo "==== Test results ===="
for H in $HORIZONS; do
    printf "%-5s " "$H"
    grep "^Test" "${LOG_DIR}/ettm1_${TAG}_${H}.log" | tail -1 || echo "(no result)"
done
