# -*- coding: utf-8 -*-
"""
PROOF that the event FRAME informationally subsumes the event POINT CLOUD for action recognition,
i.e. I(Y ; point | frame) ≈ 0. Three converging tests on frozen frameW / pointW features (SeAct base classes):

(A) Conditional complementarity — on samples FRAME gets WRONG, can POINT beat chance? (and vice versa)
    point useless on frame's errors + frame fixes point's errors  =>  frame ⊋ point.
(B) Representation similarity — linear CKA(frame_feat, point_feat).
(C) Joint upper bound — train a nonlinear MLP head on concat[frame,point] vs frame-only (frozen feats).
    no gain => point adds nothing about Y given frame, even with a strong head.

Falsification: if (A) point >> chance on frame's errors, OR (C) concat >> frame  => point NOT subsumed.
Uses existing b2n checkpoints (frameW_s*, pointW_s*). New file.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v2 import PAFClipPointV2
from paf.train_paf_b2n import split_base_new

ROOT = "/mnt/e/datasets/SeAct/preprocessed"
LOG = "paf/b2n_logs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(ckpt, lora_r, classnames, prompts):
    m = PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder="pointmamba",
                       fusion="gated", text="hand", lora_r=lora_r).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def feats(model_f, model_p, loader, base_cols, to_base):
    """Return frame_feat [N,d], point_feat [N,d], label-local [N], text_base [d,|base|]."""
    F_, P_, L_ = [], [], []
    tb = model_f._text_feats().t()[:, base_cols]                    # [d, |base|] frozen
    for frames, points, label in loader:
        frames, points = frames.to(DEV), points.to(DEV)
        F_.append(F.normalize(model_f._img_tokens(frames).mean(1), dim=-1).cpu())
        P_.append(F.normalize(model_p.point(points), dim=-1).cpu())
        L_.append(to_base[label])
    return torch.cat(F_), torch.cat(P_), torch.cat(L_), tb.cpu()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    num = (Y.t() @ X).pow(2).sum()
    den = (X.t() @ X).pow(2).sum().sqrt() * (Y.t() @ Y).pow(2).sum().sqrt()
    return (num / den.clamp_min(1e-9)).item()


def train_head(Xtr, ytr, Xva, yva, epochs=200):
    d, K = Xtr.shape[1], int(max(ytr.max(), yva.max()) + 1)
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, K)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr, ytr, Xva, yva = Xtr.to(DEV), ytr.to(DEV), Xva.to(DEV), yva.to(DEV)
    best = 0.0
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = F.cross_entropy(net(Xtr), ytr); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xva).argmax(1) == yva).float().mean().item()
        best = max(best, acc)
    return best * 100


def main():
    names = json.load(open("paf/SeAct_classes.json"))
    classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
    prompts = [f"a photo of a person {n}" for n in classnames]

    fc_all, pc_all = [], []                       # (A) pooled frame/point correctness
    cka_list = []                                 # (B)
    headF, headFP = [], []                        # (C)
    for S in [0, 1, 2]:
        base_ids, _ = split_base_new(classnames, 0.5, "random", S)
        bset = set(base_ids)
        base_cols = torch.tensor(base_ids, device=DEV)
        to_base = torch.full((len(classnames),), -1, dtype=torch.long)
        for j, c in enumerate(base_ids):
            to_base[c] = j
        tr, te, man = build_splits(os.path.join(ROOT, "manifest.json"), mode="exact",
                                   train_file="SeAct_train_norm.txt", val_file="SeAct_val_norm.txt")
        items = man["items"]
        tr_b = [i for i in tr if items[i]["label"] in bset]
        te_b = [i for i in te if items[i]["label"] in bset]

        def mk(idx):
            ds = PAFEvents(ROOT, idx, man, T=8, num_points=4096, train=False, denoise=True, frame_repr="hist")
            return DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        mf = load(f"{LOG}/frameW_s{S}.pth", 4, classnames, prompts)
        mp = load(f"{LOG}/pointW_s{S}.pth", 0, classnames, prompts)

        Fv, Pv, Lv, tb = feats(mf, mp, mk(te_b), base_cols, to_base)
        Ft, Pt, Lt, _ = feats(mf, mp, mk(tr_b), base_cols, to_base)

        # (A) predictions over base classes
        f_pred = (Fv @ tb).argmax(1); p_pred = (Pv @ tb).argmax(1)
        fc = (f_pred == Lv); pc = (p_pred == Lv)
        fc_all.append(fc); pc_all.append(pc)
        # (B)
        cka_list.append(linear_cka(Fv, Pv))
        # (C)
        headF.append(train_head(Ft, Lt, Fv, Lv))
        headFP.append(train_head(torch.cat([Ft, Pt], 1), Lt, torch.cat([Fv, Pv], 1), Lv))
        print(f"  split {S}: frame {fc.float().mean()*100:.1f}  point {pc.float().mean()*100:.1f}  "
              f"CKA {cka_list[-1]:.3f}  head(F) {headF[-1]:.1f}  head(F+P) {headFP[-1]:.1f}")

    fc = torch.cat(fc_all); pc = torch.cat(pc_all)
    chance = 100.0 / 29
    fw = ~fc; pw = ~pc
    print("\n================ SUBSUMPTION PROOF (SeAct base, 29-way, chance=%.1f%%) ================" % chance)
    print(f"overall: frame {fc.float().mean()*100:.1f}%   point {pc.float().mean()*100:.1f}%   (N={len(fc)})")
    print("\n(A) CONDITIONAL COMPLEMENTARITY")
    print(f"  point acc on FRAME's errors : {pc[fw].float().mean()*100:.1f}%  (N={int(fw.sum())}, chance {chance:.1f}%)"
          f"   <- ≈chance => point can't fix what frame misses")
    print(f"  frame acc on POINT's errors : {fc[pw].float().mean()*100:.1f}%  (N={int(pw.sum())}, chance {chance:.1f}%)"
          f"   <- ≫chance => frame fixes what point misses")
    print(f"  => frame {'⊋' if fc[pw].float().mean() > 0.3 and pc[fw].float().mean() < 0.15 else '?'} point")
    print(f"\n(B) linear CKA(frame_feat, point_feat) = {np.mean(cka_list):.3f} ± {np.std(cka_list):.3f}")
    print(f"\n(C) JOINT NONLINEAR HEAD (frozen feats):")
    print(f"  frame-only head : {np.mean(headF):.1f} ± {np.std(headF):.1f}")
    print(f"  concat[F+P] head: {np.mean(headFP):.1f} ± {np.std(headFP):.1f}   "
          f"(Δ {np.mean(headFP)-np.mean(headF):+.1f} => point's add given frame)")


if __name__ == "__main__":
    main()
