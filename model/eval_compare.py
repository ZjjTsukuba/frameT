# -*- coding: utf-8 -*-
"""
Per-class comparison: frame-only vs cross-fusion on the same val set.
Shows which actions appearance(frame-only) fails on and how much the motion+cross-attn recovers
-> the mechanistic evidence that our method "solves the frame-only problem".
New file.
"""
import os, sys, json, argparse, collections
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v2 import PAFClipPointV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--train-file", default="PAF_train.txt")
    ap.add_argument("--val-file", default="PAF_val.txt")
    ap.add_argument("--frame-ckpt", required=True)
    ap.add_argument("--cross-ckpt", required=True)
    ap.add_argument("--point-enc", default="pointmamba")
    ap.add_argument("--text", default="coop")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    tr, te, man = build_splits(os.path.join(a.root, "manifest.json"), mode="exact",
                               train_file=a.train_file, val_file=a.val_file)
    if a.classes:
        names = json.load(open(a.classes))
        if isinstance(names, dict):
            names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
        classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
        prompts = [f"a photo of a person {n}" for n in classnames]
    else:
        inv = {v: k for k, v in man["label_map"].items()}
        classnames = [inv[i].replace("-", " ") for i in range(len(inv))]
        prompts = [f"a person {c}" for c in classnames]

    model = PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder=a.point_enc,
                           fusion="cross", text=a.text, lora_r=a.lora_r).to(dev)
    val = PAFEvents(a.root, te, man, T=a.T, train=False, denoise=a.denoise)
    vl = DataLoader(val, batch_size=16, num_workers=2)

    def run(ckpt, branch):
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"], strict=False)
        model.eval(); P, L = [], []
        with torch.no_grad():
            for f, pt, l in vl:
                P += model(f.to(dev), pt.to(dev), branch).argmax(1).cpu().tolist()
                L += l.tolist()
        return np.array(P), np.array(L)

    Pf, L = run(a.frame_ckpt, "frame")
    Pc, _ = run(a.cross_ckpt, "both")
    C = len(classnames)

    def per(P):
        return {c: (P[L == c] == c).mean() for c in range(C) if (L == c).sum()}
    fa, ca = per(Pf), per(Pc)
    print(f"overall  frame-only={ (Pf==L).mean():.4f}   cross={ (Pc==L).mean():.4f}")
    rec = sorted(((ca[c] - fa[c], c) for c in fa), reverse=True)
    print("\n=== biggest recoveries (cross - frame), per class ===")
    for d, c in rec[:15]:
        print(f"  {classnames[c]:22s} frame={fa[c]:.2f} -> cross={ca[c]:.2f}  (+{d:.2f})")
    print("\n=== frame-only top confusions (true -> wrongpred : count) ===")
    conf = collections.Counter((L[i], Pf[i]) for i in range(len(L)) if L[i] != Pf[i])
    for (l, p), n in conf.most_common(12):
        print(f"  {classnames[l]:20s} -> {classnames[p]:20s} : {n}")


if __name__ == "__main__":
    main()
