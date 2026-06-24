# -*- coding: utf-8 -*-
"""
PAF dataset over preprocessed per-clip event npz. Builds ON-THE-FLY:
  (1) T polarity frames -> CLIP-ready tensor [T, 3, S, S]   (appearance branch)
  (2) an event point cloud [N, 4] = (x_n, y_n, t_n, p)      (motion branch)
All representation knobs (T, num_points, img_size, split) live here, so they are
tunable without re-running preprocessing. New file — touches no other paper's code.
"""
import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _norm_subject(s):
    # 'chnejieneng' is a typo of 'chenjieneng' in the raw filenames
    return {"chnejieneng": "chenjieneng"}.get(s, s)


def build_splits(manifest_path, mode="exact", test_ratio=0.2, test_subjects=None,
                 seed=0, split_dir=None, train_file="PAF_train.txt", val_file="PAF_val.txt"):
    """Returns (train_indices, test_indices, manifest).
    mode='exact'   -> ExACT's released PAF_train.txt / PAF_val.txt (default; comparable to ExACT 94.83%)
    mode='subject' -> leave-out test_subjects (cross-subject)
    mode='random'  -> stratified random per class
    """
    man = json.load(open(manifest_path))
    items = man["items"]
    if mode == "exact":
        if split_dir is None:
            split_dir = os.path.dirname(os.path.abspath(__file__))
        base2idx = {it["file"]: it["idx"] for it in items}

        def _read(fn):
            out = []
            for line in open(os.path.join(split_dir, fn)):
                line = line.strip().replace("\\", "/")
                if not line:
                    continue
                b = os.path.basename(line)
                if b in base2idx:
                    out.append(base2idx[b])
            return sorted(set(out))

        tr, te = _read(train_file), _read(val_file)
    elif mode == "subject":
        ts = set(test_subjects or [])
        tr = [it["idx"] for it in items if _norm_subject(it["subject"]) not in ts]
        te = [it["idx"] for it in items if _norm_subject(it["subject"]) in ts]
    else:  # stratified random per class
        rng = random.Random(seed)
        by = {}
        for it in items:
            by.setdefault(it["label"], []).append(it["idx"])
        tr, te = [], []
        for _, idxs in sorted(by.items()):
            idxs = sorted(idxs); rng.shuffle(idxs)
            k = max(1, int(round(len(idxs) * test_ratio)))
            te += idxs[:k]; tr += idxs[k:]
    return sorted(tr), sorted(te), man


class PAFEvents(Dataset):
    def __init__(self, root, indices, manifest, T=8, num_points=4096,
                 img_size=224, clip_norm=True, train=True, augment=True, denoise=False,
                 frame_repr="hist", afe_root=None):
        self.root = root
        self.items = [manifest["items"][i] for i in indices]
        self.W, self.H = manifest["W"], manifest["H"]
        self.T, self.N, self.S = T, num_points, img_size
        self.clip_norm, self.train = clip_norm, train
        self.augment = augment
        self.denoise = denoise
        self.frame_repr = frame_repr
        self.afe_root = afe_root
        self.label_map = manifest.get("label_map")
        self.num_classes = len(self.label_map) if self.label_map else manifest.get("num_classes", 0)

    def _denoise(self, x, y, t, p, sx=4, st=20000, k=2):
        """Spatiotemporal density filter: drop events whose (sx·sx px, st µs) bin has < k events."""
        if len(t) == 0:
            return x, y, t, p
        nbx = self.W // sx + 1
        bid = ((t // st) * (self.H // sx + 1) + (y // sx)) * nbx + (x // sx)
        _, inv, cnt = np.unique(bid, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= k
        if int(keep.sum()) < 100:                       # safety: never nuke the whole clip
            return x, y, t, p
        return x[keep], y[keep], t[keep], p[keep]

    def __len__(self):
        return len(self.items)

    def _load(self, it):
        d = np.load(os.path.join(self.root, it["npz"]))
        x = np.clip(d["x"].astype(np.int64), 0, self.W - 1)        # guard flaky /mnt/e reads
        y = np.clip(d["y"].astype(np.int64), 0, self.H - 1)
        t = np.clip(d["t"].astype(np.int64), 0, None)
        p = (d["p"].astype(np.int64) > 0).astype(np.int64)         # force polarity {0,1}
        return x, y, t, p, int(d["label"])

    def _frames_tsurf(self, x, y, t, p):
        """Per-window time-surface: pixel = exp(-(t_end - t_last)/tau) recency, per polarity (R=ON,B=OFF)."""
        T, H, W = self.T, self.H, self.W
        if len(t) == 0:
            return torch.zeros(T, 3, self.S, self.S)
        edges = np.linspace(0, int(t.max()) + 1, T + 1)
        surf = np.zeros((T, 2, H, W), np.float32)
        for i in range(T):
            w0, w1 = edges[i], edges[i + 1]
            m = (t >= w0) & (t < w1)
            if not m.any():
                continue
            tau = max((w1 - w0) * 0.5, 1.0)
            last = np.full((2, H, W), -np.inf)
            np.maximum.at(last, (p[m], y[m], x[m]), t[m])      # latest event time per pixel/polarity
            val = np.exp(-(w1 - last) / tau)
            val[~np.isfinite(last)] = 0.0
            surf[i] = val.astype(np.float32)
        rgb = np.stack([surf[:, 1], np.zeros_like(surf[:, 1]), surf[:, 0]], 1)
        ten = torch.nn.functional.interpolate(torch.from_numpy(rgb), size=(self.S, self.S),
                                               mode="bilinear", align_corners=False)
        if self.clip_norm:
            ten = (ten - CLIP_MEAN) / CLIP_STD
        return ten.float()

    def _frames(self, x, y, t, p):
        if self.frame_repr == "tsurf":
            return self._frames_tsurf(x, y, t, p)
        T, H, W = self.T, self.H, self.W
        tmax = int(t.max()) if len(t) else 0
        if len(t) == 0:
            return torch.zeros(T, 3, self.S, self.S)
        bins = (np.minimum((t.astype(np.float64) / (tmax + 1) * T).astype(np.int64), T - 1)
                if tmax > 0 else np.zeros(len(t), np.int64))
        fr = np.zeros((T, 2, H, W), np.float32)
        np.add.at(fr, (bins, p, y, x), 1.0)
        on, off = fr[:, 1], fr[:, 0]  # p=1 ON, p=0 OFF

        def norm(a):  # per-frame robust normalize to [0,1]
            out = np.empty_like(a)
            for i in range(a.shape[0]):
                ai = a[i]
                v = np.percentile(ai[ai > 0], 99) if (ai > 0).any() else 1.0
                out[i] = np.clip(ai / max(v, 1.0), 0, 1)
            return out

        on_n, off_n = norm(on), norm(off)
        if self.frame_repr == "histt":                   # supplement each frame with INTRA-FRAME event timing (G channel)
            pos = (t.astype(np.float64) / (tmax + 1) * T) if tmax > 0 else np.zeros(len(t))
            frac = np.clip(pos - bins, 0.0, 1.0).astype(np.float32)   # event position within its frame window [0,1)
            tsum = np.zeros((T, H, W), np.float32); tcnt = np.zeros((T, H, W), np.float32)
            np.add.at(tsum, (bins, y, x), frac)
            np.add.at(tcnt, (bins, y, x), 1.0)
            tmean = tsum / np.maximum(tcnt, 1.0)          # per-pixel mean intra-frame time (0 where no events)
            rgb = np.stack([on_n, tmean, off_n], 1)       # R=ON count, G=intra-frame timing, B=OFF count
        elif self.frame_repr == "histw":                 # ExACT-style: WHITE bg, events darken (ON->red, OFF->blue)
            rgb = np.clip(np.stack([1 - off_n, 1 - on_n - off_n, 1 - on_n], 1), 0, 1)
        else:                                            # black bg, ON->R, OFF->B
            rgb = np.stack([on_n, np.zeros_like(on_n), off_n], 1)
        ten = torch.from_numpy(rgb.astype(np.float32))
        ten = torch.nn.functional.interpolate(ten, size=(self.S, self.S),
                                               mode="bilinear", align_corners=False)
        if self.clip_norm:
            ten = (ten - CLIP_MEAN) / CLIP_STD
        return ten.float()  # [T,3,S,S]

    def _points(self, x, y, t, p):
        n, N = len(t), self.N
        if n == 0:
            return torch.zeros(N, 4)
        if n >= N:
            idx = (np.sort(np.random.choice(n, N, replace=False)) if self.train
                   else np.linspace(0, n - 1, N).astype(np.int64))
        else:
            idx = np.concatenate([np.arange(n), np.random.choice(n, N - n, replace=True)])
        tmax = max(int(t.max()), 1)
        pts = np.stack([x[idx] / self.W, y[idx] / self.H, t[idx] / tmax,
                        p[idx].astype(np.float32) * 2 - 1], 1).astype(np.float32)  # p in {-1,+1}
        return torch.from_numpy(pts)  # [N,4]

    def _augment(self, x, y, t, p):
        """Event-level augmentation (train only): temporal crop + h-flip + spatial jitter."""
        tmax = int(t.max()) if len(t) else 0
        if tmax > 0:                                    # temporal crop: 70-100% contiguous window
            win = max(int(tmax * np.random.uniform(0.7, 1.0)), 1)
            start = np.random.randint(0, max(tmax - win, 1))
            m = (t >= start) & (t < start + win)
            if int(m.sum()) > 100:
                x, y, t, p = x[m], y[m], t[m] - start, p[m]
        if np.random.rand() < 0.5:                      # horizontal flip
            x = (self.W - 1) - x
        dx, dy = np.random.randint(-15, 16), np.random.randint(-15, 16)  # mild spatial jitter
        x = np.clip(x + dx, 0, self.W - 1)
        y = np.clip(y + dy, 0, self.H - 1)
        return x, y, t, p

    def _load_afe(self, it):                            # precomputed ExACT AFE frames [T,S,S,3]
        fr = np.load(os.path.join(self.afe_root, it["npz"]))["frames"].astype(np.float32) / 255.0
        ten = torch.from_numpy(fr).permute(0, 3, 1, 2)  # [T,3,S,S]
        if self.clip_norm:
            ten = (ten - CLIP_MEAN) / CLIP_STD
        return ten.float()

    def __getitem__(self, i):
        it = self.items[i]
        x, y, t, p, label = self._load(it)
        if self.denoise:                                # cleans points (AFE frames have own denoise)
            x, y, t, p = self._denoise(x, y, t, p)
        if self.train and self.augment and self.frame_repr != "afe":
            x, y, t, p = self._augment(x, y, t, p)      # event-aug skipped for precomputed AFE frames
        frames = self._load_afe(it) if self.frame_repr == "afe" else self._frames(x, y, t, p)
        return frames, self._points(x, y, t, p), label


if __name__ == "__main__":  # quick self-test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/datasets/PAF/preprocessed")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--viz", default="/tmp/paf_ds_frames.png")
    a = ap.parse_args()
    tr, te, man = build_splits(os.path.join(a.root, "manifest.json"), mode="exact")
    print(f"split=ExACT -> train {len(tr)} test {len(te)} | classes {len(man['label_map'])}")
    ds = PAFEvents(a.root, tr, man, T=a.T, num_points=a.N, clip_norm=False, train=True)
    f, pts, lab = ds[0]
    inv = {v: k for k, v in man["label_map"].items()}
    print(f"item0: frames {tuple(f.shape)} points {tuple(pts.shape)} label {lab}({inv[lab]})")
    print(f"points x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"t[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] p∈{sorted(set(pts[:,3].tolist()))[:3]}")
    # montage of T frames (R=ON,B=OFF)
    try:
        from PIL import Image
        T = f.shape[0]
        mont = (f.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # [T,S,S,3]
        strip = np.concatenate([mont[i] for i in range(T)], axis=1)
        Image.fromarray(strip).save(a.viz)
        print("saved frame montage ->", a.viz)
    except Exception as e:
        print("viz skipped:", e)
