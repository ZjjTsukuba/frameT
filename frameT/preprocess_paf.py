# -*- coding: utf-8 -*-
"""
PAF (DAVIS346, AEDAT 2.0) preprocessing -> compact per-clip event npz.

Representation-AGNOSTIC on purpose: this only decodes raw .aedat into compact
per-clip events (x,y,t,p). Polarity frames and point clouds are built on-the-fly
in the dataset, so T / num_points / frame-format / split all stay tunable WITHOUT
re-running this step.

PAF = Miao et al. event Action Recognition, 10 classes, recorded with DAVIS346 (346x260).
AEDAT2.0 jAER-DAVIS address layout (verified on-disk):
    x   = (addr >> 12) & 0x3FF
    y   = (addr >> 22) & 0x1FF
    pol = (addr >> 11) & 1
    ts  = second int32 (microseconds, monotonic)
New file — does not touch any pre-existing dataset/model code.
"""
import os, io, json, argparse, glob
import numpy as np

W, H = 346, 260  # DAVIS346


def parse_aedat2_davis(path):
    raw = open(path, "rb").read()
    # AEDAT2.0 header = leading lines starting with b'#'; data begins after the
    # last such line. First data byte is the MSB of a DAVIS address (!= '#'),
    # so reading lines until a non-'#' line robustly finds the data offset.
    bio = io.BytesIO(raw)
    off = 0
    while True:
        p = bio.tell()
        l = bio.readline()
        if not l or not l.startswith(b"#"):
            off = p
            break
    n4 = (len(raw) - off) // 8 * 2  # whole 8-byte events -> number of >u4 elements
    d = np.frombuffer(raw, dtype=">u4", offset=off, count=n4).reshape(-1, 2)
    addr = d[:, 0].astype(np.int64)
    ts = d[:, 1].astype(np.int64)
    x = ((addr >> 12) & 0x3FF).astype(np.int32)
    y = ((addr >> 22) & 0x1FF).astype(np.int32)
    p = ((addr >> 11) & 1).astype(np.int8)
    # keep only in-bounds DVS events (drop any stray APS/IMU/special addresses)
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, p, ts = x[m], y[m], p[m], ts[m]
    # unwrap 32-bit microsecond timestamp wraparound: a ~5 s clip can straddle the
    # 2^32 us boundary; without this the span looks like ~4295 s and sorting scrambles it.
    if len(ts) > 1:
        dd = np.diff(ts)
        cum = np.concatenate([[0], np.cumsum((dd < -(2 ** 31)).astype(np.int64) * (2 ** 32))])
        ts = ts + cum
    if len(ts) and not np.all(np.diff(ts) >= 0):
        o = np.argsort(ts, kind="stable")
        x, y, p, ts = x[o], y[o], p[o], ts[o]
    t = (ts - ts.min()).astype(np.int64) if len(ts) else ts
    return x, y, t, p, off


def subsample(x, y, t, p, cap, rng):
    n = len(t)
    if cap and n > cap:
        idx = np.sort(rng.choice(n, cap, replace=False))
        return x[idx], y[idx], t[idx], p[idx]
    return x, y, t, p


def main():
    ap = argparse.ArgumentParser("PAF AEDAT2.0 -> per-clip event npz")
    ap.add_argument("--src", default="/mnt/e/datasets/PAF/Action Recognition")
    ap.add_argument("--out", default="/mnt/e/datasets/PAF/preprocessed")
    ap.add_argument("--cap", type=int, default=400000, help="max events/clip kept (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    classes = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    label_map = {c: i for i, c in enumerate(classes)}
    print("classes:", label_map)

    manifest, k = [], 0
    for c in classes:
        files = sorted(glob.glob(os.path.join(args.src, c, "*.aedat")))
        for fp in files:
            name = os.path.basename(fp)
            subj = name.split("_")[0]
            try:
                x, y, t, p, hdr = parse_aedat2_davis(fp)
            except Exception as e:
                print("FAIL", fp, e)
                continue
            n0 = len(t)
            if n0 == 0:
                print("EMPTY", fp)
                continue
            x, y, t, p = subsample(x, y, t, p, args.cap, rng)
            outp = os.path.join(args.out, f"{k:04d}.npz")
            np.savez_compressed(
                outp,
                x=x.astype(np.uint16), y=y.astype(np.uint16),
                t=t.astype(np.uint32), p=p.astype(np.uint8),
                label=np.int64(label_map[c]),
            )
            manifest.append(dict(idx=k, npz=os.path.basename(outp), file=name,
                                 clazz=c, label=label_map[c], subject=subj,
                                 n_events=int(len(t)), n_raw=int(n0),
                                 dur_us=int(t.max()), hdr_bytes=int(hdr)))
            k += 1
            if k % 20 == 0:
                print(f"  [{k:3d}] {c}/{name}  events={len(t)} (raw {n0})  hdr={hdr}")
    json.dump(dict(label_map=label_map, W=W, H=H, cap=args.cap, items=manifest),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    subj = sorted(set(m["subject"] for m in manifest))
    print(f"done: {k} clips -> {args.out}/manifest.json | subjects({len(subj)}): {subj}")


if __name__ == "__main__":
    main()
