# Event-Camera Action Recognition with CLIP-LoRA + FrameTemporalMamba

Code for our ACCV 2026 submission on **event-stream human action recognition**.

A deliberately simple frame-based pipeline — frozen CLIP ViT-B/32 adapted with a
small LoRA, a learnable CoOp text prompt, and an order-aware **FrameTemporalMamba**
over event-histogram frames — **matches or exceeds** the heavier ExACT (CVPR 2024)
pipeline (AFE + cross-modal conceptual reasoning + uncertainty) on its own splits,
without AFE, captioning, or conceptual-reasoning modules.

> The core narrative is *subtraction*: once appearance is properly adapted (regularized
> LoRA on plain histogram frames), motion/point-cloud branches, finer temporal
> granularity, white-bg coloring, time-surfaces, deeper fusion, and bigger adapters are
> all redundant or harmful on these small event-action sets. The point-cloud / GNN
> exploration code is **not** part of this paper and lives elsewhere.

## Results

Closed-set top-1 accuracy on the **ExACT official splits** (same train/val lists, 80/20).

| Dataset | Split (train/val) | Method | Top-1 |
|---|---|---|---|
| **SeAct** (58 cls) | 464 / 116 | ExACT-caption (CVPR'24) | 67.24 (Top-5 75.00) |
| **SeAct** | 464 / 116 | **Ours — frameT (3-seed mean)** | **75.58** (77.59 / 75.00 / 74.14) |
| **PAF** (10 cls) | 234 / 58 | ExACT (CVPR'24) | 94.83 |
| **PAF** | 234 / 58 | **Ours — frameT (3-seed mean)** | **93.1** (frameM mean-pool 91.9; TemporalMamba +1.2) |

Headline: on SeAct our **Top-1 (75.58)** exceeds ExACT's **Top-5 (75.00)** — one guess
beats its five — with a far lighter model.

## Method

Event stream → **histogram frames** (T=8, black background, ON/OFF event-count channels)
→ spatiotemporal-density **denoise** + event-level **augmentation** (temporal crop, h-flip,
jitter; train only) → **CLIP ViT-B/32** image tower (frozen) + **LoRA** (r=4, on q/k/v/out
of all 12 ViT layers) → per-frame [CLS] tokens → **FrameTemporalMamba** (causal, order-aware
over the T frame tokens; falls back to a causal BiGRU if `mamba_ssm` is absent) → cosine
similarity to a **CoOp** learnable text prompt.

## Repository layout

```
paf/
  preprocess_paf.py      PAF  (AEDAT 2.0, self-contained decoder) -> per-clip npz
  preprocess_seact.py    SeAct (AEDAT4 via dv_processing)         -> per-clip npz
  paf_dataset.py         PAFEvents: T histogram frames + [N,4] point cloud,
                         denoise / augment / hist|histw|tsurf|afe|histt, build_splits
  model_v1.py            point encoders (PointNet/PointNet++/PointMamba), CrossAttn/Deep fusion
  model_v2.py            PAFClipPointV2: CLIP + LoRA + CoOp + GatedCrossAttnFusion + FrameTemporalMamba
  train_paf.py           main training entry point
  train_paf_b2n.py       base-to-new / open-vocabulary protocol
  eval_compare.py        per-class frame-only vs fused confusion analysis
  per_class_temporal.py  per-class TemporalMamba effect
  make_llm_text_emb.py   precompute CLIP text embeddings (template / LLM kinematic anchors)
  afe.py, preprocess_afe.py   faithful ExACT-AFE port (ablation; underperforms our histograms)
  alpha_sweep_b2n.py, proof_subsumption.py   analysis scripts
  run_*.sh               multi-seed training sweeps (TemporalMamba, T=16, b2n, ablations)
  *_train.txt / *_val.txt     ExACT official splits (SeAct *_norm = de-prefixed, 464/116)
  SeAct_classes.json, SeAct_idx_to_label.json, seact_prompts_llm.json   class maps + LLM anchors
  *_text_emb*.pt         precomputed CLIP text embeddings
```

Model checkpoints (the `frameT_s{0,1,2}` SOTA seeds and ablations) are **not committed**;
they are available on request.

## Setup

```bash
pip install -r requirements.txt
# optional, for the exact SOTA temporal model (else a BiGRU fallback is used):
# pip install mamba-ssm causal-conv1d
```

Run everything as a module from the repository root so package imports resolve:

```bash
python -m paf.train_paf ...        # not: python paf/train_paf.py
```

## Reproduce

**1. Preprocess** events into per-clip npz caches (point them at your local dataset roots):

```bash
python -m paf.preprocess_paf      # -> <PAF_root>/preprocessed/*.npz   + manifest.json
python -m paf.preprocess_seact    # -> <SeAct_root>/preprocessed/*.npz + manifest.json
```

**2. Train the SOTA `frameT` config** (frozen CLIP + LoRA r4 + CoOp + FrameTemporalMamba, T=8):

SeAct (58 classes, ExACT split):
```bash
python -m paf.train_paf \
  --root <SeAct_root>/preprocessed --classes paf/SeAct_classes.json \
  --train-file SeAct_train_norm.txt --val-file SeAct_val_norm.txt \
  --branch frame --text coop --lora-r 4 --lr 1e-4 \
  --denoise --temporal-mamba --T 8 --epochs 60 --seed 0
```

PAF (10 classes, ExACT split; labels come from the manifest, no `--classes`):
```bash
python -m paf.train_paf \
  --root <PAF_root>/preprocessed \
  --train-file PAF_train.txt --val-file PAF_val.txt \
  --branch frame --text coop --lora-r 4 --lr 1e-4 \
  --denoise --temporal-mamba --T 8 --epochs 60 --seed 0
```

Multi-seed sweeps and all ablations are scripted in `paf/run_*.sh`
(`run_temporal_sweep.sh` = SeAct seeds, `run_temporal_paf.sh` = PAF seeds).
`frameM` (drop `--temporal-mamba`) is the order-blind mean-pool baseline;
`--branch both --fusion gated` adds the (redundant) point-cloud motion branch.

## Key ablations (all in this repo)

- **Motion is redundant over good appearance.** Always-on point-cloud fusion is net-0 (PAF)
  to −3.5 (SeAct) once LoRA is trained; gated fusion recovers it only to ≈+0.9 (within noise).
- **Appearance adapters overfit on small data.** LoRA r8, MLP adapter, deep fusion, per-layer
  injection, more frames (T=16), and uncapped event counts all hurt; only denoise + regularized
  LoRA (r4, lr 1e-4) help.
- **Language ceiling ≈ 1–2 pts.** A fixed LLM kinematic-description anchor adds +1.72 over a bare
  template (matching ExACT's caption +1.17), yet a content-free learnable CoOp prompt beats both
  — prompt *learnability* > text semantic richness.
- **Open-vocabulary (base→new).** `train_paf_b2n.py`: LoRA visual-domain adaptation transfers to
  unseen classes (NEW 7.5 → 34.0, H 6.9 → 46.7).

## Datasets

- **SeAct** — DVS semantic action, 58 classes, DAVIS346, AEDAT4. ExACT split (464/116).
- **PAF** — 10 classes, DAVIS346, AEDAT 2.0. ExACT split (234/58).

We use ExACT's released train/val lists and its closed-set top-1 protocol for a clean,
controlled comparison.

## License

MIT — see [LICENSE](LICENSE).
