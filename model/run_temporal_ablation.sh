#!/bin/bash
# Firm up the TemporalMamba gain: 2x2 ablation (frame/gated x mean/temporal) + more seeds.
#   frameM = frame-only mean-pool (baseline ~69.8) | frameT = frame + TemporalMamba (no point)
#   base   = gated mean-pool (have s0-2)           | temp   = gated + TemporalMamba (have s0-2)
# Key Qs: (1) frameT>frameM => temporal helps without point; (2) temp~frameT => point redundant given temporal.
set -u
ROOT=/mnt/e/datasets/SeAct/preprocessed
LOG=paf/temporal_logs
mkdir -p "$LOG"
COM="--root $ROOT --classes paf/SeAct_classes.json \
--train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
--epochs 60 --T 8 --point-enc pointmamba --text coop --denoise --lora-r 4 --lr 1e-4 --fusion gated --workers 2"
run() { local tag=$1; shift; python3 -m paf.train_paf $COM "$@" \
        --out "$LOG/$tag.pth" > "$LOG/$tag.log" 2>&1; echo "DONE $tag"; }

# frame-only (NO point) ablation, 3 seeds: mean vs temporal
for S in 0 1 2; do
  run frameM_s$S --branch frame --seed $S &
  run frameT_s$S --branch frame --seed $S --temporal-mamba &
  wait
done
# +3 more seeds for gated base vs temp (firm up the +4.6 headline)
for S in 3 4 5; do
  run base_s$S --branch both --seed $S &
  run temp_s$S --branch both --seed $S --temporal-mamba &
  wait
done
echo "ALL ABLATION DONE"
