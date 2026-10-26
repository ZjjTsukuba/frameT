# Event-Camera Action Recognition with CLIP-LoRA + FrameTemporalMamba

Code for my graduation research (Master's thesis, University of Tsukuba) on **event-stream human action recognition**.

A deliberately simple frame-based pipeline — frozen CLIP ViT-B/32 adapted with a
small LoRA, a learnable CoOp text prompt, and an order-aware **FrameTemporalMamba**
over event-histogram frames — evaluated on SeAct, PAF (ExACT official splits) and
DailyDVS-200, without AFE, captioning, or conceptual-reasoning modules.

> The core narrative is *subtraction*: once appearance is properly adapted (regularized
> LoRA on plain histogram frames), motion/point-cloud branches, finer temporal
> granularity, white-bg coloring, time-surfaces, deeper fusion, and bigger adapters are
> not needed on these event-action sets. The point-cloud / GNN
> exploration code is **not** part of this thesis and lives elsewhere.

## Results

Quantitative results will be added after the thesis is published.

Evaluation protocol: closed-set top-1 accuracy. SeAct / PAF use the ExACT official train/val
lists; DailyDVS-200 uses its official subject-disjoint split (model selection on val, test on
`DailyDVS200_test_clean.txt` = official test minus the clips duplicated in val).

## Method

Event stream → **histogram frames** (T=8, black background, ON/OFF event-count channels)
→ spatiotemporal-density **denoise** + event-level **augmentation** (temporal crop, h-flip,
jitter; train only) → **CLIP ViT-B/32** image tower (frozen) + **LoRA** (r=4, on q/k/v/out
of all 12 ViT layers) → per-frame [CLS] tokens → **FrameTemporalMamba** (causal, order-aware
over the T frame tokens; falls back to a causal BiGRU if `mamba_ssm` is absent) → cosine
similarity to a **CoOp** learnable text prompt.

## Repository layout

```
frameT/
  preprocess_paf.py      PAF  (AEDAT 2.0, self-contained decoder) -> per-clip npz
  preprocess_seact.py    SeAct (AEDAT4 via dv_processing)         -> per-clip npz
  preprocess_dailydvs200.py  DailyDVS-200 (200 cls, AEDAT4/DVXplorer Lite 320x240, 22046 clips,
                         subject-disjoint train/val/test) -> per-clip npz; official splits +
                         class list in DailyDVS200_*.{txt,json}; DailyDVS200_test_clean.txt =
                         test minus clips duplicated in val
  event_dataset.py         EventClipDataset: T histogram frames + [N,4] point cloud,
                         denoise / augment / hist|histw|tsurf|afe|histt, build_splits
  model_v1.py            point encoders (PointNet/PointNet++/PointMamba), CrossAttn/Deep fusion
  model_v2.py            FrameTModel: CLIP + LoRA + CoOp + GatedCrossAttnFusion + FrameTemporalMamba
  train.py           main training entry point
  train_b2n.py       base-to-new / open-vocabulary protocol
  eval_compare.py        per-class frame-only vs fused confusion analysis
  per_class_temporal.py  per-class TemporalMamba effect
  make_llm_text_emb.py   precompute CLIP text embeddings (template / LLM kinematic anchors)
  afe.py, preprocess_afe.py   faithful ExACT-AFE port (ablation)
  alpha_sweep_b2n.py, proof_subsumption.py   analysis scripts
  run_*.sh               multi-seed training sweeps (TemporalMamba, T=16, b2n, ablations)
  *_train.txt / *_val.txt     ExACT official splits (SeAct *_norm = de-prefixed, 464/116)
  SeAct_classes.json, SeAct_idx_to_label.json, seact_prompts_llm.json   class maps + LLM anchors
  *_text_emb*.pt         precomputed CLIP text embeddings
```

Model checkpoints are **not committed**;
they are available on request.

## Setup

```bash
pip install -r requirements.txt
# optional, for the Mamba temporal model (else a BiGRU fallback is used):
# pip install mamba-ssm causal-conv1d
```

Run everything as a module from the repository root so package imports resolve:

```bash
python -m frameT.train ...        # not: python frameT/train.py
```

## Reproduce

**1. Preprocess** events into per-clip npz caches (point them at your local dataset roots):

```bash
python -m frameT.preprocess_paf      # -> <PAF_root>/preprocessed/*.npz   + manifest.json
python -m frameT.preprocess_seact    # -> <SeAct_root>/preprocessed/*.npz + manifest.json
python -m frameT.preprocess_dailydvs200 --zip <DailyDvs-200.zip> --out <DailyDVS_root>/preprocessed
```

**2. Train the `frameT` config** (frozen CLIP + LoRA r4 + CoOp + FrameTemporalMamba, T=8):

SeAct (58 classes, ExACT split):
```bash
python -m frameT.train \
  --root <SeAct_root>/preprocessed --classes frameT/SeAct_classes.json \
  --train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
  --branch frame --frame-head none --text coop --ctx-init "a photo of a person" --lora-r 4 --lr 1e-4 \
  --denoise --temporal-mamba --T 8 --epochs 60 --seed 0
```

PAF (10 classes, ExACT split; labels come from the manifest, no `--classes`):
```bash
python -m frameT.train \
  --root <PAF_root>/preprocessed \
  --train-file PAF_train.txt --val-file PAF_val.txt \
  --branch frame --frame-head none --text coop --ctx-init "a photo of a person" --lora-r 4 --lr 1e-4 \
  --denoise --temporal-mamba --T 8 --epochs 60 --seed 0
```

Add `--frame-repr histt` for the in-frame timestamp representation.

DailyDVS-200 (200 classes, official subject-disjoint split; model selection on val, test on the
val-overlap-free `DailyDVS200_test_clean.txt`; use `DailyDVS200_test.txt` for the full official test):
```bash
python -m frameT.train   --root <DailyDVS_root>/preprocessed --classes frameT/DailyDVS200_classes.json   --train-file DailyDVS200_train.txt --val-file DailyDVS200_val.txt --test-file DailyDVS200_test_clean.txt   --branch frame --frame-head none --text coop --ctx-init "a photo of a person" --lora-r 4 --lr 1e-4   --denoise --temporal-mamba --frame-repr histt --T 8 --epochs 12 --workers 6 --seed 0
```

Useful extras: `--resume` (per-epoch checkpoint `<out>.ckpt`, continue after an interruption),
`--dump-preds <file.npz>` (save val logits with original / reversed / shuffled frame order at each new
best epoch; analyse with `python frameT/analyze_preds.py`), `--temporal-kind {mamba,tcn,lstm,gru,transformer,s4}`
and `--temporal-gate {auto,on,off}` (temporal-model comparison).

Multi-seed sweeps and all ablations are scripted in `frameT/run_*.sh`
(`run_temporal_sweep.sh` = SeAct seeds, `run_temporal_paf.sh` = PAF seeds).
`frameM` (drop `--temporal-mamba`) is the order-blind mean-pool baseline;
`--branch both --fusion gated` adds the point-cloud motion branch.

## Design notes

- **Frame-only, no head between CLIP and the text (`--frame-head none`, the final recipe).** The
  legacy default (`--fusion concat`, `--frame-head mlp`) passes the frame embedding through a randomly
  initialized 2-layer MLP (a leftover of the point branch) before the text cosine. This discards
  CLIP's image-text alignment at init and can collapse training to chance depending on the data
  order. Comparing the pooled CLIP embedding with the text directly avoids this, with one default
  temperature (tau0 2.659) for all datasets.
- **Fixed-template baseline:** use `--text hand_openai` (not `hand`, which builds V1 without LoRA/Mamba).
- **CoOp context init (`--ctx-init "a photo of a person"`).** The 5 context vectors are initialized
  from the template's word embeddings, so training starts exactly at the template and stays learnable.
- **Uniform time bins, no interpolation.** `--time-bins count` and `--interp` are kept as options
  but are not part of the final recipe.
- **LoRA scale.** Keep alpha / r = 4 (e.g. r 4 / alpha 16); a much larger scale can destabilize
  training of the frozen-CLIP features.
- **Optional modules kept for ablations (off by default):** point-cloud branch and fusion
  (`--branch both --fusion gated`), `--patch-pool`, `--bbox-crop`, ExACT-AFE frames (`afe.py`),
  other temporal models (`--temporal-kind`), base-to-new protocol (`train_b2n.py`).

## Datasets

- **SeAct** — DVS semantic action, 58 classes, DAVIS346, AEDAT4. ExACT split (464/116).
- **PAF** — 10 classes, DAVIS346, AEDAT 2.0. ExACT split (234/58).
- **DailyDVS-200** — 200 classes, DVXplorer Lite, AEDAT4. Official subject-disjoint split (14,773 / 3,735 / 4,093).

For SeAct and PAF we use ExACT's released train/val lists and its closed-set top-1 protocol.

## License

MIT — see [LICENSE](LICENSE).
