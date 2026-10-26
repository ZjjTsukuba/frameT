# -*- coding: utf-8 -*-
"""Test-time augmentation (horizontal flip averaging) on the existing baseline seed0 checkpoint
(94.83%, already the ExACT-matching best) -- cheap, no retraining -- to see whether it can rescue
the 2 persistently-misclassified 'waving' samples (53, 56), which are wrong across every
architecture/preprocessing variant tried so far (patch-pool, bbox-crop)."""
import os, sys
import torch
import torch.nn.functional as F
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

val_ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True)
vl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

model = FrameTModel(classnames, hand_prompts=prompts, point_encoder="pointnet2",
                        fusion="concat", text="coop", lora_r=4, temporal_mamba=True).to(device)
ckpt = torch.load(os.path.expanduser("~/research/results/paf_frameT_s0.pth"), map_location=device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

preds_plain, preds_tta, trues = [], [], []
with torch.no_grad():
    for frames, points, label in vl:
        frames = frames.to(device); points = points.to(device)
        logits_orig = model(frames, points, "frame")
        logits_flip = model(torch.flip(frames, dims=[-1]), points, "frame")   # horizontal flip (W axis)
        probs = F.softmax(logits_orig, -1) + F.softmax(logits_flip, -1)
        preds_plain.extend(logits_orig.argmax(1).cpu().tolist())
        preds_tta.extend(probs.argmax(1).cpu().tolist())
        trues.extend(label.tolist())

acc_plain = sum(a == b for a, b in zip(preds_plain, trues)) / len(trues)
acc_tta = sum(a == b for a, b in zip(preds_tta, trues)) / len(trues)
print(f"plain acc={acc_plain:.4f}  TTA(flip-avg) acc={acc_tta:.4f}")
for i in [53, 56]:
    print(f"  idx {i}: true={classnames[trues[i]]} plain_pred={classnames[preds_plain[i]]} tta_pred={classnames[preds_tta[i]]}")
print("=== TTA misclassified ===")
for i in range(len(trues)):
    if preds_tta[i] != trues[i]:
        print(f"  idx {i}: true={classnames[trues[i]]:20s} pred={classnames[preds_tta[i]]}")
