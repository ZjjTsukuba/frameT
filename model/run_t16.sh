#!/bin/bash
# Finer-temporal closing test: T=16 frames + TemporalMamba (vs T=8 frameT 75.58).
# More bins → finer time-steps for the SSM (≠ histt's intra-bin channel). 3 seeds, SeAct, frame-only. New file.
set -u
ROOT=/mnt/e/datasets/SeAct/preprocessed
LOG=paf/temporal_logs
COM="--root $ROOT --classes paf/SeAct_classes.json \
--train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
--epochs 60 --T 16 --branch frame --point-enc pointmamba --text coop --denoise \
--lora-r 4 --lr 1e-4 --fusion gated --temporal-mamba --frame-repr hist --workers 2"
run() { local tag=$1; shift; python3 -m paf.train_paf $COM "$@" \
        --out "$LOG/$tag.pth" > "$LOG/$tag.log" 2>&1; echo "DONE $tag"; }

run t16_s0 --seed 0 &
run t16_s1 --seed 1 &
wait
run t16_s2 --seed 2 &
wait
echo "ALL T16 DONE"
