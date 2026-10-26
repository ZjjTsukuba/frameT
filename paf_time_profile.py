# -*- coding: utf-8 -*-
"""Check event-rate-over-time profile for the 2 persistently-misclassified 'waving' samples vs
a correctly-classified one, to test whether the action is temporally concentrated near the end
of a long recording (uniform T=8 time-binning would then waste most frames on near-silence)."""
import os, sys
import numpy as np
sys.path.append(os.path.expanduser("~/research/frameT"))
from frameT.event_dataset import build_splits, EventClipDataset

ROOT = os.path.expanduser("~/research/data/PAF/preprocessed")
tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                            train_file="PAF_train.txt", val_file="PAF_val.txt")
ds = EventClipDataset(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True)

for i, tag in [(53, "hard1"), (56, "hard2"), (52, "correct")]:
    it = ds.items[i]
    x, y, t, p, label = ds._load(it)
    xk, yk, tk, pk = ds._denoise(x, y, t, p)
    tmax = int(tk.max()) if len(tk) else 0
    nbins = 20
    edges = np.linspace(0, tmax + 1, nbins + 1)
    counts, _ = np.histogram(tk, bins=edges)
    frac = counts / max(counts.sum(), 1)
    bar = "".join("#" if f > 0.02 else ("." if f > 0.002 else " ") for f in frac)
    print(f"{tag} (idx {i}, {it['file']}, dur={tmax/1e6:.2f}s, n={len(tk)}): [{bar}]")
    print(f"  frac per bin (20 bins over duration): {[round(f,3) for f in frac]}")
