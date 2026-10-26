# -*- coding: utf-8 -*-
"""Visualize frame montages for the 2 persistently-misclassified 'waving' val samples (53,56)
vs. a correctly-classified 'waving' sample, to inspect whether the motion visually resembles
throwing/turning (systematic confusion pairs found in the confusion analysis)."""
import os, sys
import numpy as np
from PIL import Image
sys.path.append(os.path.expanduser("~/research/frameT"))
from frameT.event_dataset import build_splits, EventClipDataset

ROOT = os.path.expanduser("~/research/data/PAF/preprocessed")
tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                            train_file="PAF_train.txt", val_file="PAF_val.txt")
inv = {v: k for k, v in man["label_map"].items()}
ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True, clip_norm=False, bbox_crop=True)

waving_idx = [i for i in range(len(ds.items)) if ds.items[i] and int(np.load(os.path.join(ROOT, ds.items[i]["npz"]))["label"]) == man["label_map"]["waving"]]
print("all waving val indices:", waving_idx)

targets = {53: "hard1", 56: "hard2"}
correct_candidate = [i for i in waving_idx if i not in (53, 56)][0]
targets[correct_candidate] = "correct_example"

out_dir = os.path.expanduser("~/research/results/waving_viz_bboxcrop")
os.makedirs(out_dir, exist_ok=True)
for i, tag in targets.items():
    f, pts, lab = ds[i]
    print(f"idx {i} ({tag}): label={lab}({inv[lab]}) file={ds.items[i]['file']}")
    mont = (f.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    strip = np.concatenate([mont[t] for t in range(mont.shape[0])], axis=1)
    path = os.path.join(out_dir, f"{tag}_idx{i}.png")
    Image.fromarray(strip).save(path)
    print("  saved ->", path)
