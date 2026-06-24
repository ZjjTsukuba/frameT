# -*- coding: utf-8 -*-
"""Per-class accuracy: gated base (mean-pool) vs +TemporalMamba on SeAct val (58-way).
Tests whether TemporalMamba's gain concentrates on temporally-confusable (order-reversed) actions.
Loads checkpoints one at a time (low mem, can run alongside training). New file."""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v2 import PAFClipPointV2

ROOT = "/mnt/e/datasets/SeAct/preprocessed"
LOG = "paf/temporal_logs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# temporally-confusable / order-reversed pairs to watch
WATCH = ["open the computer", "close the computer", "put on glasses", "put off glasses",
         "lift the box", "put down the box", "catch and throw a ball", "catch a ball",
         "throw a ball", "sit down", "squat down", "stand"]


@torch.no_grad()
def per_class(ckpt, temporal, classnames, prompts, loader):
    m = PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder="pointmamba",
                       fusion="gated", text="coop", lora_r=4, temporal_mamba=temporal).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
    m.eval()
    C = len(classnames)
    cor = np.zeros(C); tot = np.zeros(C)
    for frames, points, label in loader:
        pred = m(frames.to(DEV), points.to(DEV), "both").argmax(1).cpu().numpy()
        lab = label.numpy()
        for p, l in zip(pred, lab):
            tot[l] += 1; cor[l] += int(p == l)
    del m; torch.cuda.empty_cache()
    return cor, tot


def main():
    names = json.load(open("paf/SeAct_classes.json"))
    classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
    prompts = [f"a photo of a person {n}" for n in classnames]
    tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                               train_file="SeAct_train_norm.txt", val_file="SeAct_val_norm.txt")
    ds = PAFEvents(ROOT, te, man, T=8, num_points=4096, train=False, denoise=True, frame_repr="hist")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    bc, bt = per_class(f"{LOG}/base_s0.pth", False, classnames, prompts, loader)
    tc, tt = per_class(f"{LOG}/temp_s0.pth", True, classnames, prompts, loader)
    base_acc = bc / np.maximum(bt, 1); temp_acc = tc / np.maximum(tt, 1)
    print(f"overall: base {bc.sum()/bt.sum()*100:.1f}%  temp {tc.sum()/tt.sum()*100:.1f}%  (val={int(bt.sum())})")

    delta = temp_acc - base_acc
    order = np.argsort(-delta)
    print("\n=== biggest TEMPORAL gains (temp - base), per class ===")
    for i in order[:12]:
        if bt[i] == 0: continue
        star = " *WATCH" if classnames[i] in WATCH else ""
        print(f"  {classnames[i]:28s} base {base_acc[i]:.2f} -> temp {temp_acc[i]:.2f}  ({delta[i]:+.2f}){star}")
    print("\n=== biggest TEMPORAL losses ===")
    for i in order[-6:]:
        if bt[i] == 0: continue
        print(f"  {classnames[i]:28s} base {base_acc[i]:.2f} -> temp {temp_acc[i]:.2f}  ({delta[i]:+.2f})")
    print("\n=== WATCH (temporally-confusable/order-reversed) classes ===")
    for i, c in enumerate(classnames):
        if c in WATCH and bt[i] > 0:
            print(f"  {c:28s} base {base_acc[i]:.2f} -> temp {temp_acc[i]:.2f}  ({delta[i]:+.2f})  (n={int(bt[i])})")


if __name__ == "__main__":
    main()
