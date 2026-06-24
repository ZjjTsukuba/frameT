# -*- coding: utf-8 -*-
"""
Train PAF V1: frozen CLIP frame + PointNet point + concat fusion -> CLIP text. Metric: top-1.
Protocol: ExACT's released PAF split (PAF_train.txt / PAF_val.txt) -> comparable to ExACT 94.83%.
New file.
"""
import os, sys, argparse, json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v1 import PAFClipPoint

# natural-language phrasing for CLIP text prototypes
PHRASE = {
    "arm crossing": "crossing their arms", "get-up": "getting up", "jumping": "jumping",
    "kicking": "kicking", "picking up": "picking something up", "sit-down": "sitting down",
    "throwing": "throwing", "turning around": "turning around", "walking": "walking",
    "waving": "waving",
}


def prompts_from(label_map):
    inv = {v: k for k, v in label_map.items()}
    return [f"a person {PHRASE.get(inv[i], inv[i].replace('-', ' '))}" for i in range(len(inv))]


@torch.no_grad()
def evaluate(model, loader, device, branch="both"):
    model.eval()
    correct = total = 0
    for frames, points, label in loader:
        pred = model(frames.to(device), points.to(device), branch).argmax(1).cpu()
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser("PAF V1 training")
    ap.add_argument("--root", default="/mnt/e/datasets/PAF/preprocessed")
    ap.add_argument("--classes", default=None, help="json list/dict of class names (e.g. SeAct_classes.json)")
    ap.add_argument("--train-file", default="PAF_train.txt")
    ap.add_argument("--val-file", default="PAF_val.txt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="paf/paf_v1_best.pth")
    ap.add_argument("--branch", default="both", choices=["both", "frame", "point"])
    ap.add_argument("--point-enc", default="pointnet2", choices=["pointnet", "pointnet2", "pointmamba", "tfirst"])
    ap.add_argument("--fusion", default="concat", choices=["concat", "cross", "deep", "gated"])
    ap.add_argument("--text", default="hand", choices=["hand", "coop", "hand_openai"])
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=0, help=">0 enables LoRA on CLIP ViT")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--img-adapter", action="store_true", help="MLP feature adapter on frozen CLIP image feats")
    ap.add_argument("--denoise", action="store_true", help="spatiotemporal density denoise (frames+points)")
    ap.add_argument("--inject-every", type=int, default=0, help=">0: inject motion into ViT every k layers (needs LoRA)")
    ap.add_argument("--temporal-mamba", action="store_true", help="order-aware TemporalMamba over T frame tokens (vs mean-pool)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (model init + data shuffle)")
    ap.add_argument("--prompt-file", default=None, help="json {classname: description}; replaces the 'a photo of a person X' template anchors with richer text (e.g. LLM kinematic descriptions). Use with --text hand.")
    ap.add_argument("--frame-repr", default="hist", choices=["hist", "histw", "tsurf", "afe", "histt"])
    ap.add_argument("--afe-root", default=None, help="dir of precomputed AFE frames (for --frame-repr afe)")
    args = ap.parse_args()

    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    import numpy as _np, random as _r
    torch.manual_seed(args.seed); _np.random.seed(args.seed); _r.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tr, te, man = build_splits(os.path.join(args.root, "manifest.json"), mode="exact",
                               train_file=args.train_file, val_file=args.val_file)
    n_cls = len(man['label_map']) if man.get('label_map') else man.get('num_classes', 0)
    print(f"[data] train {len(tr)} val {len(te)} classes {n_cls}")
    train_ds = PAFEvents(args.root, tr, man, T=args.T, num_points=args.points, train=True,
                         denoise=args.denoise, frame_repr=args.frame_repr, afe_root=args.afe_root)
    val_ds = PAFEvents(args.root, te, man, T=args.T, num_points=args.points, train=False,
                       denoise=args.denoise, frame_repr=args.frame_repr, afe_root=args.afe_root)
    pw = args.workers > 0
    tl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=pw)
    vl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=pw)

    if args.classes:
        names = json.load(open(args.classes))
        if isinstance(names, dict):
            names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
        classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
        prompts = [f"a photo of a person {n}" for n in classnames]
        if args.prompt_file:                       # richer text anchors (e.g. LLM descriptions)
            desc = json.load(open(args.prompt_file))
            prompts = [desc[n] for n in classnames]
            print(f"[prompts] LLM/file anchors from {args.prompt_file}; sample: {prompts[0]}")
    else:
        prompts = prompts_from(man["label_map"])
        inv = {v: k for k, v in man["label_map"].items()}
        classnames = [inv[i].replace("-", " ") for i in range(len(inv))]
    print(f"[classes] {len(classnames)} | sample prompts: {prompts[:3]}")
    if args.text in ("coop", "hand_openai"):
        from paf.model_v2 import PAFClipPointV2
        model = PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder=args.point_enc,
                               fusion=args.fusion, text=("coop" if args.text == "coop" else "hand"),
                               n_ctx=args.n_ctx, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                               img_adapter=args.img_adapter, inject_every=args.inject_every,
                               temporal_mamba=args.temporal_mamba).to(device)
    else:
        model = PAFClipPoint(prompts, point_encoder=args.point_enc, fusion=args.fusion).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = torch.nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train(); model.clip.eval()    # CLIP stays frozen/eval
        tot = n = 0
        for frames, points, label in tl:
            frames, points, label = frames.to(device), points.to(device), label.to(device)
            loss = lossf(model(frames, points, args.branch), label)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * label.numel(); n += label.numel()
        sched.step()
        acc = evaluate(model, vl, device, args.branch)
        if acc >= best:
            best = acc
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            sd = {k: v for k, v in model.state_dict().items()
                  if k in trainable or not k.startswith("clip.")}
            torch.save({"model": sd, "acc": acc, "epoch": ep, "prompts": prompts}, args.out)
        print(f"ep {ep + 1:3d}/{args.epochs}  loss {tot / max(n, 1):.4f}  val-top1 {acc:.4f}  best {best:.4f}")

    print(f"\n[done] best val top-1 = {best:.4f}   (ExACT reports 94.83% on this split)")


if __name__ == "__main__":
    main()
