#!/bin/bash
# Idempotent resume of the PAF temporal 2x2 sweep: skip runs whose log already has "[done]".
# (4/12 finished before the GPU was freed: frameM/frameT s0,s1.) New file.
set -u
ROOT=/mnt/e/datasets/PAF/preprocessed
LOG=paf/temporal_paf_logs
mkdir -p "$LOG"
COM="--root $ROOT --train-file PAF_train.txt --val-file PAF_val.txt \
--epochs 60 --T 8 --point-enc pointmamba --text coop --denoise \
--lora-r 4 --lr 1e-4 --fusion gated --workers 2"
run() { local tag=$1; shift
  if grep -q "\[done\]" "$LOG/$tag.log" 2>/dev/null; then echo "SKIP $tag (already done)"; return; fi
  python3 -m paf.train_paf $COM "$@" --out "$LOG/$tag.pth" > "$LOG/$tag.log" 2>&1
  echo "DONE $tag"; }

for S in 0 1 2; do
  run frameM_s$S --branch frame --seed $S &
  run frameT_s$S --branch frame --seed $S --temporal-mamba &
  wait
done
for S in 0 1 2; do
  run base_s$S --branch both --seed $S &
  run temp_s$S --branch both --seed $S --temporal-mamba &
  wait
done
echo "ALL PAF TEMPORAL DONE"
