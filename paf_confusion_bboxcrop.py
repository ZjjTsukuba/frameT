# -*- coding: utf-8 -*-
"""Confusion check for the bbox-crop seed0 checkpoint: does it fix the persistently-misclassified
waving samples (53, 56), and what's the overall per-class picture vs the baseline seed0?"""
import os, sys
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

val_ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True, bbox_crop=True)
vl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

model = FrameTModel(classnames, hand_prompts=prompts, point_encoder="pointnet2",
                        fusion="concat", text="coop", lora_r=4, temporal_mamba=True).to(device)
ckpt = torch.load(os.path.expanduser("~/research/results/paf_bboxcrop_s0.pth"), map_location=device)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print(f"missing(non-clip)={[k for k in missing if not k.startswith('clip.')]} unexpected={unexpected}")
model.eval()

preds, trues = [], []
with torch.no_grad():
    for frames, points, label in vl:
        out = model(frames.to(device), points.to(device), "frame")
        preds.extend(out.argmax(1).cpu().tolist())
        trues.extend(label.tolist())
acc = sum(a == b for a, b in zip(preds, trues)) / len(trues)
print(f"reproduced acc={acc:.4f} (ckpt stored={ckpt['acc']:.4f})")

print("=== per-class accuracy ===")
for c in range(len(classnames)):
    idx = [i for i, t in enumerate(trues) if t == c]
    if not idx:
        continue
    correct = sum(preds[i] == c for i in idx)
    print(f"{classnames[c]:20s} (n={len(idx)}): {correct}/{len(idx)}")

print("=== samples 53, 56 (the persistent hard waving ones) ===")
for i in [53, 56]:
    print(f"  idx {i}: true={classnames[trues[i]]} pred={classnames[preds[i]]}")

print("=== all misclassified ===")
for i in range(len(trues)):
    if preds[i] != trues[i]:
        print(f"  idx {i}: true={classnames[trues[i]]:20s} pred={classnames[preds[i]]}")
