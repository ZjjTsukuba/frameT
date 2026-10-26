# -*- coding: utf-8 -*-
"""Aggregate confusion analysis across the 3 baseline PAF frameT checkpoints (s0,s1,s2).
Loads each checkpoint into the exact training architecture, runs eval, records per-sample
predictions, and reports: per-class accuracy, and aggregated confusion pairs across the 3 seeds
-- to see whether errors are concentrated in specific class pairs (systematic) or spread out
(noise from the small 58-sample val set)."""
import os, sys, json
import torch
sys.path.append(os.path.expanduser("~/research/frameT"))
from frameT.event_dataset import build_splits, EventClipDataset
from frameT.model_v2 import FrameTModel
from frameT.train import prompts_from
from torch.utils.data import DataLoader

ROOT = os.path.expanduser("~/research/data/PAF/preprocessed")
device = torch.device("cuda")

tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                            train_file="PAF_train.txt", val_file="PAF_val.txt")
inv = {v: k for k, v in man["label_map"].items()}
classnames = [inv[i].replace("-", " ") for i in range(len(inv))]
prompts = prompts_from(man["label_map"])
n_cls = len(classnames)

val_ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True)
vl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

per_sample_true = None
per_seed_preds = {}
per_seed_acc = {}

for seed in [0, 1, 2]:
    ckpt_path = os.path.expanduser(f"~/research/results/paf_frameT_s{seed}.pth")
    model = FrameTModel(classnames, hand_prompts=prompts, point_encoder="pointnet2",
                            fusion="concat", text="coop", lora_r=4, temporal_mamba=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  ckpt keys={len(ckpt['model'])} missing={len(missing)} unexpected={len(unexpected)}")
    non_clip_missing = [k for k in missing if not k.startswith("clip.")]
    print(f"  NON-clip-prefixed missing keys ({len(non_clip_missing)}): {non_clip_missing}")
    print(f"  unexpected sample: {unexpected[:5]}")
    print(f"  ckpt keys list (all {len(ckpt['model'])}): {sorted(ckpt['model'].keys())}")
    print(f"  has prompt_learner.ctx in ckpt: {'prompt_learner.ctx' in ckpt['model']}")
    print(f"  ckpt logit_scale key present: {'logit_scale' in ckpt['model']}, model.logit_scale={model.logit_scale.item():.4f}")
    lora_key = "clip.vision_model.encoder.layers.0.self_attn.q_proj.A"
    if lora_key in ckpt["model"]:
        print(f"  ckpt {lora_key} norm={ckpt['model'][lora_key].norm().item():.4f}")
    sd_now = dict(model.named_parameters())
    if lora_key in sd_now:
        print(f"  loaded model {lora_key} norm={sd_now[lora_key].norm().item():.4f}")
    print(f"  ckpt prompt_learner.ctx norm={ckpt['model']['prompt_learner.ctx'].norm().item():.4f}")
    print(f"  loaded model prompt_learner.ctx norm={dict(model.named_parameters())['prompt_learner.ctx'].norm().item():.4f}")
    for k in ["temporal.seq.A_log", "temporal.seq.out_proj.weight", "temporal.norm.weight"]:
        in_ckpt = k in ckpt["model"]
        ckpt_norm = ckpt["model"][k].norm().item() if in_ckpt else None
        model_norm = dict(model.named_parameters())[k].norm().item() if k in dict(model.named_parameters()) else None
        print(f"  {k}: in_ckpt={in_ckpt} ckpt_norm={ckpt_norm} model_norm={model_norm}")
    model.eval()

    with torch.no_grad():
        tf = model._text_feats()          # [C,dim] normalized
        sim = tf @ tf.t()
        off_diag = sim - torch.diag(torch.diag(sim))
        print(f"  text feat pairwise cos-sim: diag(mean)={torch.diag(sim).mean().item():.4f} "
              f"off-diag(mean)={off_diag.sum().item()/(n_cls*(n_cls-1)):.4f} off-diag(max)={off_diag.max().item():.4f}")

    preds, trues = [], []
    first = True
    with torch.no_grad():
        for frames, points, label in vl:
            frames_d = frames.to(device)
            out = model(frames_d, points.to(device), "frame")
            if first:
                print(f"  frames stats: mean={frames_d.mean().item():.4f} std={frames_d.std().item():.4f} "
                      f"min={frames_d.min().item():.4f} max={frames_d.max().item():.4f} shape={tuple(frames_d.shape)}")
                print(f"  logits[0]={out[0].detach().cpu().tolist()}")
                print(f"  logits stats: mean={out.mean().item():.4f} std={out.std().item():.4f}")
                import torch.nn.functional as _F
                img_feat = _F.normalize(model._img_tokens(frames_d).mean(1), dim=-1)   # [B,dim]
                isim = img_feat @ img_feat.t()
                ioff = isim - torch.diag(torch.diag(isim))
                print(f"  image feat pairwise cos-sim across batch: off-diag mean={ioff.sum().item()/(ioff.shape[0]*(ioff.shape[0]-1)):.4f} "
                      f"min={ioff[ioff!=0].min().item():.4f} max={ioff.max().item():.4f}")
                print(f"  batch labels: {label.tolist()}")
                first = False
            preds.extend(out.argmax(1).cpu().tolist())
            trues.extend(label.tolist())
    per_seed_preds[seed] = preds
    if per_sample_true is None:
        per_sample_true = trues
    acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
    per_seed_acc[seed] = acc
    print(f"seed {seed}: reproduced eval acc = {acc:.4f} (ckpt stored acc = {ckpt['acc']:.4f})")

print()
print("=== per-class accuracy per seed (support in parens) ===")
n_val = len(per_sample_true)
for c in range(n_cls):
    idx = [i for i, t in enumerate(per_sample_true) if t == c]
    if not idx:
        continue
    row = []
    for seed in [0, 1, 2]:
        correct = sum(per_seed_preds[seed][i] == c for i in idx)
        row.append(f"{correct}/{len(idx)}")
    print(f"{classnames[c]:20s} (n={len(idx)}): " + "  ".join(row))

print()
print("=== misclassified samples (true -> pred) aggregated across all 3 seeds ===")
from collections import Counter
conf = Counter()
for seed in [0, 1, 2]:
    for i in range(n_val):
        t, p = per_sample_true[i], per_seed_preds[seed][i]
        if t != p:
            conf[(classnames[t], classnames[p])] += 1
for (t, p), cnt in conf.most_common():
    print(f"{t:20s} -> {p:20s}: {cnt} (out of 3 seeds)")

print()
print("=== samples wrong in ALL 3 seeds (persistently hard) ===")
for i in range(n_val):
    t = per_sample_true[i]
    wrong_all = all(per_seed_preds[seed][i] != t for seed in [0, 1, 2])
    if wrong_all:
        preds_str = ", ".join(classnames[per_seed_preds[seed][i]] for seed in [0, 1, 2])
        print(f"sample {i}: true={classnames[t]} | preds(s0,s1,s2)={preds_str}")
