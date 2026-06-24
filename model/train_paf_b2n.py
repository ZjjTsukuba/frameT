# -*- coding: utf-8 -*-
"""
Base-to-new (open-vocabulary) event action recognition on SeAct.
Classes are split into BASE (trained) and NEW (held out, never seen in training).
We train on base-class samples only; at eval we classify base-test among BASE names
and new-test among NEW names (disjoint label spaces) -> report base / new / H (harmonic mean),
the standard CoOp/CoCoOp base-to-new protocol.

Crux question this answers:
  (1) frozen CLIP   -> zero-shot floor on event frames (no training).
  (2) LoRA on base  -> does adapting the ViT to base classes HELP or KILL new-class transfer?
  (3) +gated motion -> does the point-cloud motion cue help NEW (unseen) classes?

The model (PAFClipPointV2) outputs logits over ALL classes; base/new is done purely by
slicing logit columns + remapping labels here, so the model is untouched. New file.
"""
import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v2 import PAFClipPointV2


def split_base_new(classnames, base_frac=0.5, mode="alpha", seed=0):
    """Deterministic class split. 'alpha' = CoOp-style alphabetical halves; 'random' = seeded."""
    n = len(classnames)
    nb = int(round(n * base_frac))
    if mode == "alpha":
        order = sorted(range(n), key=lambda i: classnames[i])
    else:
        order = list(range(n))
        import random as _r
        _r.Random(seed).shuffle(order)
    return sorted(order[:nb]), sorted(order[nb:])


@torch.no_grad()
def eval_subset(model, loader, device, cols, to_local, branch):
    """Accuracy restricted to `cols` (a class-id subset); labels remapped via `to_local`."""
    model.eval()
    correct = total = 0
    for frames, points, label in loader:
        logits = model(frames.to(device), points.to(device), branch)      # [B, C_all]
        pred_local = logits[:, cols].argmax(1).cpu()                       # index into the subset
        target_local = to_local[label]                                     # base/new-local gt
        correct += (pred_local == target_local).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser("PAF/SeAct base-to-new (open-vocab) training")
    ap.add_argument("--root", default="/mnt/e/datasets/SeAct/preprocessed")
    ap.add_argument("--classes", default="paf/SeAct_classes.json")
    ap.add_argument("--train-file", default="SeAct_train_norm.txt")
    ap.add_argument("--val-file", default="SeAct_val_norm.txt")
    ap.add_argument("--base-frac", type=float, default=0.5)
    ap.add_argument("--split-mode", default="alpha", choices=["alpha", "random"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="training seed (model init + data shuffle)")
    ap.add_argument("--epochs", type=int, default=60, help="0 = eval-only (frozen zero-shot floor)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-point", type=float, default=None, help="separate lr for point encoder (dual): from-scratch likes 1e-3")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--branch", default="frame", choices=["both", "frame", "point", "dual"])
    ap.add_argument("--point-enc", default="pointmamba", choices=["pointnet", "pointnet2", "pointmamba"])
    ap.add_argument("--fusion", default="gated", choices=["concat", "cross", "deep", "gated"])
    ap.add_argument("--text", default="hand", choices=["hand", "coop"])
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--out", default="paf/seact_b2n.pth")
    args = ap.parse_args()

    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- class names + base/new split ----
    names = json.load(open(args.classes))
    if isinstance(names, dict):
        names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
    classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
    prompts = [f"a photo of a person {n}" for n in classnames]
    base_ids, new_ids = split_base_new(classnames, args.base_frac, args.split_mode, args.split_seed)
    base_set, new_set = set(base_ids), set(new_ids)
    print(f"[split] {len(classnames)} classes -> base {len(base_ids)} / new {len(new_ids)} ({args.split_mode})")
    print(f"[base] {[classnames[i] for i in base_ids]}")
    print(f"[new ] {[classnames[i] for i in new_ids]}")

    # column selectors + label->local-index maps (size = #classes; -1 where out of subset)
    base_cols = torch.tensor(base_ids, device=device)
    new_cols = torch.tensor(new_ids, device=device)
    to_base = torch.full((len(classnames),), -1, dtype=torch.long)
    to_new = torch.full((len(classnames),), -1, dtype=torch.long)
    for j, c in enumerate(base_ids):
        to_base[c] = j
    for j, c in enumerate(new_ids):
        to_new[c] = j

    # ---- data: train=base-only, val split into base / new ----
    tr, te, man = build_splits(os.path.join(args.root, "manifest.json"), mode="exact",
                               train_file=args.train_file, val_file=args.val_file)
    items = man["items"]
    tr_base = [i for i in tr if items[i]["label"] in base_set]
    te_base = [i for i in te if items[i]["label"] in base_set]
    te_new = [i for i in te if items[i]["label"] in new_set]
    print(f"[data] train(base) {len(tr_base)}  val-base {len(te_base)}  val-new {len(te_new)}")

    def mk(idx, train):
        ds = PAFEvents(args.root, idx, man, T=args.T, num_points=args.points, train=train,
                       denoise=args.denoise, frame_repr="hist")
        return DataLoader(ds, batch_size=args.bs, shuffle=train, num_workers=args.workers,
                          drop_last=train, pin_memory=True, persistent_workers=args.workers > 0)

    tl = mk(tr_base, True)
    vl_base, vl_new = mk(te_base, False), mk(te_new, False)

    # ---- model (holds ALL class prompts; base/new = column slicing) ----
    model = PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder=args.point_enc,
                           fusion=args.fusion, text=args.text, n_ctx=args.n_ctx,
                           lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M  "
          f"| lora_r={args.lora_r} text={args.text} branch={args.branch} fusion={args.fusion}")

    def report(tag, ep):
        b = eval_subset(model, vl_base, device, base_cols, to_base, args.branch)
        n = eval_subset(model, vl_new, device, new_cols, to_new, args.branch)
        h = 2 * b * n / max(b + n, 1e-9)
        print(f"{tag} ep {ep}  base {b:.4f}  new {n:.4f}  H {h:.4f}")
        return b, n, h

    if args.epochs == 0:                                   # frozen zero-shot floor
        report("[frozen]", 0)
        return

    if args.lr_point is None:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    else:                                                  # point encoder (from-scratch) gets its own lr
        pt = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("point.")]
        ot = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("point.")]
        opt = torch.optim.AdamW([{"params": ot, "lr": args.lr}, {"params": pt, "lr": args.lr_point}],
                                weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = torch.nn.CrossEntropyLoss()
    best_h = 0.0
    for ep in range(args.epochs):
        model.train(); model.clip.eval()
        tot = n = 0
        for frames, points, label in tl:
            frames, points = frames.to(device), points.to(device)
            target = to_base[label].to(device)             # base-local target
            logits = model(frames, points, args.branch)[:, base_cols]
            loss = lossf(logits, target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * label.numel(); n += label.numel()
        sched.step()
        b, nw, h = report(f"loss {tot / max(n,1):.3f} |", ep + 1)
        if h >= best_h:
            best_h = h
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            sd = {k: v for k, v in model.state_dict().items()
                  if k in trainable or not k.startswith("clip.")}   # +buffers (BN running stats) of non-clip modules
            torch.save({"model": sd, "base": b, "new": nw, "H": h, "epoch": ep,
                        "base_ids": base_ids, "new_ids": new_ids}, args.out)
    print(f"\n[done] best H = {best_h:.4f}  (base-to-new harmonic mean)")


if __name__ == "__main__":
    main()
