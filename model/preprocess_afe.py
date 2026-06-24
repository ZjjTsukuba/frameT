# -*- coding: utf-8 -*-
"""
Precompute ExACT AFE frames from an event npz cache -> afe frame cache (T frames/clip),
aligned by filename so the dataset reads AFE frames (appearance) + events (motion) together.
New file.
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.afe import adaptive_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)              # event cache (manifest.json + *.npz)
    ap.add_argument("--out", required=True)               # AFE frame cache dir
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--S", type=int, default=224)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(os.path.join(a.root, "manifest.json")))["items"]
    nfr = []
    for i, it in enumerate(items):
        d = np.load(os.path.join(a.root, it["npz"]))
        ev = np.stack([d["x"], d["y"], d["t"], d["p"]], 1).astype(np.float64)  # sorted by t
        frames = np.asarray(adaptive_sample(ev, S=a.S))   # [K,S,S,3]
        K = len(frames); nfr.append(K)
        idx = (np.linspace(0, K - 1, a.T).astype(np.int64) if K >= a.T
               else np.concatenate([np.arange(K), np.full(a.T - K, K - 1, dtype=np.int64)]))
        np.savez_compressed(os.path.join(a.out, it["npz"]), frames=frames[idx].astype(np.uint8))
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(items)}] {it['npz']} AFE_frames={K}")
    print(f"done: {len(items)} clips | AFE frames/clip min/mean/max "
          f"{min(nfr)}/{int(np.mean(nfr))}/{max(nfr)} -> {a.out}")


if __name__ == "__main__":
    main()
