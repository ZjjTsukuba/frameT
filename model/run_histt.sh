#!/bin/bash
# User's idea: supplement each frame with INTRA-FRAME timing (G channel from event timestamps),
# fed through the strong CLIP+LoRA+TemporalMamba — vs plain count histogram (hist, =frameT 75.58).
# 3 seeds, frame-only. New file.
set -u
ROOT=/mnt/e/datasets/SeAct/preprocessed
LOG=paf/temporal_logs
COM="--root $ROOT --classes paf/SeAct_classes.json \
--train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
--epochs 60 --T 8 --branch frame --point-enc pointmamba --text coop --denoise \
--lora-r 4 --lr 1e-4 --fusion gated --temporal-mamba --frame-repr histt --workers 2"
run() { local tag=$1; shift; python3 -m paf.train_paf $COM "$@" \
        --out "$LOG/$tag.pth" > "$LOG/$tag.log" 2>&1; echo "DONE $tag"; }

run histt_s0 --seed 0 &
run histt_s1 --seed 1 &
wait
run histt_s2 --seed 2 &
wait
echo "ALL HISTT DONE"
