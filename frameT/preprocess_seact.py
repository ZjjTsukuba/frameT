# -*- coding: utf-8 -*-
"""
SeAct (DVS Semantic Action; 58 classes; AEDAT4 / DAVIS346) -> compact per-clip event npz,
SAME format as PAF (x,y,t,p,label) so the dataset/model/train pipeline is reused unchanged.

AEDAT4 parsed via dv_processing. Label = idx_to_label[int(filename prefix before first '-')]
(ExACT's SeAct_idx_to_label.json, 58 classes). Files live under SeAct/<subject>/<prefix>-<ts>.aedat4.
New file.
"""
import os, json, argparse, glob
import numpy as np
import dv_processing as dv

W, H = 346, 260  # DAVIS346


def parse_aedat4(path):
    r = dv.io.MonoCameraRecording(path)
    chunks = []
    while r.isRunning():
        b = r.getNextEventBatch()
        if b is not None:
            chunks.append(b.numpy())
    if not chunks:
        return None
    a = np.concatenate(chunks)
    x = a["x"].astype(np.int32); y = a["y"].astype(np.int32)
    t = a["timestamp"].astype(np.int64); p = (a["polarity"] > 0).astype(np.int8)
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t, p = x[m], y[m], t[m], p[m]
    if len(t) == 0:
        return None
    o = np.argsort(t, kind="stable")                 # ensure chronological
    x, y, t, p = x[o], y[o], t[o], p[o]
    return x, y, (t - t.min()).astype(np.int64), p


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser("SeAct AEDAT4 -> per-clip event npz")
    ap.add_argument("--src", default="/mnt/e/datasets/SeAct")
    ap.add_argument("--out", default="/mnt/e/datasets/SeAct/preprocessed")
    ap.add_argument("--idx2lab", default=os.path.join(here, "SeAct_idx_to_label.json"))
    ap.add_argument("--cap", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    idx2lab = json.load(open(args.idx2lab))
    subjects = sorted(d for d in os.listdir(args.src)
                      if os.path.isdir(os.path.join(args.src, d)) and d != "preprocessed")
    print("subjects:", subjects)

    manifest, k = [], 0
    for sub in subjects:
        for fp in sorted(glob.glob(os.path.join(args.src, sub, "*.aedat4"))):
            name = os.path.basename(fp)
            prefix = name.split("-")[0]
            key = prefix if prefix in idx2lab else str(int(prefix))
            if key not in idx2lab:
                print("NO LABEL", name); continue
            label = int(idx2lab[key])
            try:
                r = parse_aedat4(fp)
            except Exception as e:
                print("FAIL", fp, e); continue
            if r is None:
                print("EMPTY", fp); continue
            x, y, t, p = r
            n0 = len(t)
            if args.cap and n0 > args.cap:
                idx = np.sort(rng.choice(n0, args.cap, replace=False))
                x, y, t, p = x[idx], y[idx], t[idx], p[idx]
            outp = os.path.join(args.out, f"{k:04d}.npz")
            np.savez_compressed(outp, x=x.astype(np.uint16), y=y.astype(np.uint16),
                                t=t.astype(np.uint32), p=p.astype(np.uint8), label=np.int64(label))
            manifest.append(dict(idx=k, npz=os.path.basename(outp), file=name, subject=sub,
                                 label=label, prefix=prefix, n_events=int(len(t)),
                                 n_raw=int(n0), dur_us=int(t.max())))
            k += 1
            if k % 50 == 0:
                print(f"  [{k}] {sub}/{name} label={label} events={len(t)} (raw {n0})")

    labset = sorted(set(m["label"] for m in manifest))
    json.dump(dict(W=W, H=H, cap=args.cap, num_classes=58, items=manifest),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print(f"done: {k} clips, {len(labset)} classes -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
