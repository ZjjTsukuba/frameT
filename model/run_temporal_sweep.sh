#!/bin/bash
# Tonight's task: does FrameTemporalMamba (order-aware over T frames) help our gated-fusion SOTA
# (frame CLIP-LoRA + point cloud gated at the end)? Multi-seed on SeAct.
# base = gated SOTA (mean-pool, order-blind); temp = + FrameTemporalMamba. 3 seeds each, 2-wide. New file.
set -u
ROOT=/mnt/e/datasets/SeAct/preprocessed
LOG=paf/temporal_logs
mkdir -p "$LOG"
COM="--root $ROOT --classes paf/SeAct_classes.json \
--train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
--epochs 60 --T 8 --branch both --point-enc pointmamba --text coop --denoise \
--lora-r 4 --lr 1e-4 --fusion gated --workers 2"

run() { local tag=$1; shift; python3 -m paf.train_paf $COM "$@" \
        --out "$LOG/$tag.pth" > "$LOG/$tag.log" 2>&1; echo "DONE $tag"; }

for S in 0 1 2; do
  run base_s$S --seed $S &                       # gated SOTA (mean-pool)
  run temp_s$S --seed $S --temporal-mamba &       # + order-aware TemporalMamba
  wait
done
echo "ALL TEMPORAL DONE"
