# -*- coding: utf-8 -*-
"""Inspect the persistently-misclassified 'waving' val samples: raw event count, and how much
the spatiotemporal denoise filter removes, to test the hypothesis that denoise disproportionately
strips sparse/fast small-limb motion (waving) events."""
import os, sys
import numpy as np
sys.path.append(os.path.expanduser("~/research/frameT"))
from frameT.event_dataset import build_splits, EventClipDataset

ROOT = os.path.expanduser("~/research/data/PAF/preprocessed")
tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                            train_file="PAF_train.txt", val_file="PAF_val.txt")
inv = {v: k for k, v in man["label_map"].items()}

ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True)
ds_nodn = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=False)

waving_label = [k for k, v in man["label_map"].items() if "wav" in k.lower()]
print("waving label name(s):", waving_label)

for i in [53, 56]:
    it = ds.items[i]
    x, y, t, p, label = ds._load(it)
    print(f"sample {i}: file={it.get('file')} label={label}({inv[label]}) raw n_events={len(t)} duration_us={int(t.max()) if len(t) else 0}")
    xk, yk, tk, pk = ds._denoise(x, y, t, p)
    print(f"  after denoise: n_events={len(tk)} kept_frac={len(tk)/max(len(t),1):.3f}")

print()
print("=== avg kept_frac by class (denoise impact) ===")
from collections import defaultdict
sums = defaultdict(list)
for i in range(len(ds.items)):
    it = ds.items[i]
    x, y, t, p, label = ds._load(it)
    xk, yk, tk, pk = ds._denoise(x, y, t, p)
    frac = len(tk) / max(len(t), 1)
    sums[label].append((frac, len(t)))
for label, vals in sorted(sums.items()):
    fracs = [v[0] for v in vals]
    counts = [v[1] for v in vals]
    print(f"{inv[label]:20s} n={len(vals):2d}  mean_kept_frac={np.mean(fracs):.3f}  mean_raw_events={np.mean(counts):8.0f}")
