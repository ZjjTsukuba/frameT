#!/bin/bash
# Base-to-new (open-vocab) SeAct sweep: 3 random class splits x {frozen, LoRA-frame, LoRA+motion, CoOp}.
# Deterministic eval FPS (model_v1 _fps) -> numbers are stable; we report mean over splits.
# 60-epoch jobs run 2-wide. New file.
set -u
ROOT=/mnt/e/datasets/SeAct/preprocessed
LOG=paf/b2n_logs
mkdir -p "$LOG"
PY=paf/train_paf_b2n.py
COM="--root $ROOT --denoise --workers 4 --split-mode random --seed 0"

run() { local tag=$1; shift; python3 $PY $COM "$@" > "$LOG/$tag.log" 2>&1; echo "DONE $tag"; }

echo "=== frozen floors (fast) ==="
for S in 0 1 2; do
  run frozen_s$S --split-seed $S --epochs 0 --lora-r 0 --text hand --branch frame --fusion gated
done

echo "=== LoRA frame vs LoRA+motion (2-wide per split) ==="
for S in 0 1 2; do
  run loraF_s$S --split-seed $S --epochs 60 --lora-r 4 --text hand --branch frame --fusion gated --lr 1e-4 &
  run loraM_s$S --split-seed $S --epochs 60 --lora-r 4 --text hand --branch both  --fusion gated --lr 1e-4 &
  wait
done

echo "=== CoOp text-adaptation baseline (no LoRA), 2-wide ==="
run coop_s0 --split-seed 0 --epochs 60 --lora-r 0 --text coop --branch frame --fusion gated --lr 3e-4 &
run coop_s1 --split-seed 1 --epochs 60 --lora-r 0 --text coop --branch frame --fusion gated --lr 3e-4 &
wait
run coop_s2 --split-seed 2 --epochs 60 --lora-r 0 --text coop --branch frame --fusion gated --lr 3e-4 &
wait

echo "ALL DONE"
