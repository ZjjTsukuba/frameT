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
from frameT.event_dataset import build_splits, EventClipDataset
from frameT.model_v1 import FrameTModelV1

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
def evaluate(model, loader, device, branch="both", topk=False):
    model.eval()
    correct = c5 = total = 0
    for frames, points, label in loader:
        out = model(frames.to(device), points.to(device), branch)
        correct += (out.argmax(1).cpu() == label).sum().item()
        if topk:
            k = min(5, out.shape[1])
            c5 += (out.topk(k, 1).indices.cpu() == label[:, None]).any(1).sum().item()
        total += label.numel()
    if topk:
        return correct / max(total, 1), c5 / max(total, 1)
    return correct / max(total, 1)


@torch.no_grad()
def dump_orders(model, loader, device, branch, path, ep):
    """Val logits with the frames in original, reversed and randomly shuffled order (frame-order
    sensitivity test). Global RNG state is saved/restored so training is unaffected."""
    import numpy as np
    rs = torch.get_rng_state(); rc = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    g = torch.Generator().manual_seed(1234)
    model.eval()
    out = {"orig": [], "rev": [], "shuf": []}; labels = []
    for frames, points, label in loader:
        B, T = frames.shape[:2]
        perm = torch.stack([torch.randperm(T, generator=g) for _ in range(B)])
        variants = {"orig": frames, "rev": frames.flip(1), "shuf": frames[torch.arange(B)[:, None], perm]}
        for k, f in variants.items():
            out[k].append(model(f.to(device), points.to(device), branch).float().cpu().numpy())
        labels.append(label.numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["labels"] = np.concatenate(labels); res["epoch"] = np.array(ep)
    np.savez_compressed(path + ".tmp.npz", **res); os.replace(path + ".tmp.npz", path)
    print("  [orders] " + "  ".join(f"{k} {(res[k].argmax(1) == res['labels']).mean():.4f}" for k in ("orig", "rev", "shuf")), flush=True)
    torch.set_rng_state(rs)
    if rc is not None:
        torch.cuda.set_rng_state_all(rc)


def main():
    ap = argparse.ArgumentParser("frameT training")
    ap.add_argument("--root", default="/mnt/e/datasets/PAF/preprocessed")
    ap.add_argument("--classes", default=None, help="json list/dict of class names (e.g. SeAct_classes.json)")
    ap.add_argument("--train-file", default="PAF_train.txt")
    ap.add_argument("--val-file", default="PAF_val.txt")
    ap.add_argument("--test-file", default=None, help="optional held-out list: evaluated whenever val-top1 improves; reported at the best-val epoch")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="frameT/paf_v1_best.pth")
    ap.add_argument("--branch", default="both", choices=["both", "frame", "point"])
    ap.add_argument("--point-enc", default="pointnet2", choices=["pointnet", "pointnet2", "pointmamba", "tfirst"])
    ap.add_argument("--fusion", default="concat", choices=["concat", "cross", "deep", "gated"])
    ap.add_argument("--text", default="hand", choices=["hand", "coop", "hand_openai"])
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--ctx-init", default=None, help="CoOp: init the context vectors from this template's word embeddings (e.g. 'a photo of a person'); n_ctx becomes its token count")
    ap.add_argument("--lora-r", type=int, default=0, help=">0 enables LoRA on CLIP ViT")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--img-adapter", action="store_true", help="MLP feature adapter on frozen CLIP image feats")
    ap.add_argument("--denoise", action="store_true", help="spatiotemporal density denoise (frames+points)")
    ap.add_argument("--inject-every", type=int, default=0, help=">0: inject motion into ViT every k layers (needs LoRA)")
    ap.add_argument("--temporal-mamba", action="store_true", help="order-aware TemporalMamba over T frame tokens (vs mean-pool)")
    ap.add_argument("--temporal-kind", default="mamba", choices=["mamba", "gru", "lstm", "transformer", "tcn", "s4"], help="sequence model used with --temporal-mamba (comparison study; default mamba)")
    ap.add_argument("--temporal-gate", default="auto", choices=["auto", "on", "off"], help="zero-init tanh gate on the temporal branch: auto = non-mamba kinds only")
    ap.add_argument("--dump-preds", default=None, help="npz path: at each new best epoch save val labels + logits for original / reversed / shuffled frame order")
    ap.add_argument("--frame-head", default="mlp", choices=["mlp", "none"], help="frame branch + concat fusion: legacy random concat-MLP, or 'none' = CLIP embedding compared with text directly")
    ap.add_argument("--patch-pool", action="store_true", help="attention-pool ViT patch tokens (local appearance) fused with CLS")
    ap.add_argument("--seed", type=int, default=0, help="random seed (model init + data shuffle)")
    ap.add_argument("--logit-scale-init", type=float, default=None, help="initial tau of the learnable temperature s=exp(tau) (default 2.659; CLIP uses 4.605). Large class counts (e.g. 200) may need a sharper start")
    ap.add_argument("--log-every", type=int, default=0, help="print running train loss every N steps (0=off)")
    ap.add_argument("--logit-scale-lr", type=float, default=None, help="separate learning rate for tau (default: same as --lr)")
    ap.add_argument("--prompt-file", default=None, help="json {classname: description}; replaces the 'a photo of a person X' template anchors with richer text (e.g. LLM kinematic descriptions). Use with --text hand.")
    ap.add_argument("--frame-repr", default="hist", choices=["hist", "histw", "tsurf", "afe", "histt"])
    ap.add_argument("--afe-root", default=None, help="dir of precomputed AFE frames (for --frame-repr afe)")
    ap.add_argument("--time-bins", default="uniform", choices=["uniform", "count"], help="frame boundaries: equal duration (default) or equal event count (sec 3.2.5)")
    ap.add_argument("--interp", action="store_true", help="linear temporal voting into adjacent frames (sec 3.2.6)")
    ap.add_argument("--bbox-crop", action="store_true", help="crop to event spatial bbox (1/99 pct, padded) before resize -- helps small/distant subjects")
    ap.add_argument("--resume", action="store_true", help="save <out>.ckpt every epoch (model, optimizer, scheduler, best, RNG) and continue from it if it exists; removed after [done]")
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
    train_ds = EventClipDataset(args.root, tr, man, T=args.T, num_points=args.points, train=True,
                         denoise=args.denoise, frame_repr=args.frame_repr, afe_root=args.afe_root,
                         bbox_crop=args.bbox_crop,
                         time_bins=args.time_bins, interp=args.interp)
    val_ds = EventClipDataset(args.root, te, man, T=args.T, num_points=args.points, train=False,
                       denoise=args.denoise, frame_repr=args.frame_repr, afe_root=args.afe_root,
                       bbox_crop=args.bbox_crop,
                         time_bins=args.time_bins, interp=args.interp)
    pw = args.workers > 0
    tl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=pw)
    vl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=max(2, args.workers // 2),
                    pin_memory=True, persistent_workers=pw)

    tl_test = None
    if args.test_file:
        _, te_t, _ = build_splits(os.path.join(args.root, "manifest.json"), mode="exact",
                                  train_file=args.train_file, val_file=args.test_file)
        test_ds = EventClipDataset(args.root, te_t, man, T=args.T, num_points=args.points, train=False,
                            denoise=args.denoise, frame_repr=args.frame_repr, afe_root=args.afe_root,
                            bbox_crop=args.bbox_crop,
                         time_bins=args.time_bins, interp=args.interp)
        tl_test = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=max(2, args.workers // 2),
                             pin_memory=True, persistent_workers=pw)
        print(f"[data] test {len(te_t)}")

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
        from frameT.model_v2 import FrameTModel
        model = FrameTModel(classnames, hand_prompts=prompts, point_encoder=args.point_enc,
                               fusion=args.fusion, text=("coop" if args.text == "coop" else "hand"),
                               n_ctx=args.n_ctx, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                               img_adapter=args.img_adapter, inject_every=args.inject_every,
                               temporal_mamba=args.temporal_mamba, patch_pool=args.patch_pool,
                               temporal_kind=args.temporal_kind, temporal_gate=args.temporal_gate, frame_head=args.frame_head,
                               ctx_init=args.ctx_init).to(device)
    else:
        model = FrameTModelV1(prompts, point_encoder=args.point_enc, fusion=args.fusion).to(device)
    if args.logit_scale_init is not None:
        model.logit_scale.data.fill_(args.logit_scale_init)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    if args.logit_scale_lr is not None:
        rest = [p for p in params if p is not model.logit_scale]
        opt = torch.optim.AdamW([{"params": rest}, {"params": [model.logit_scale], "lr": args.logit_scale_lr, "weight_decay": 0.0}],
                                lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = torch.nn.CrossEntropyLoss()

    def trainable_state():
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        return {k: v for k, v in model.state_dict().items()
                if k in trainable or not k.startswith("clip.")}

    best = 0.0; best5 = 0.0; test_at_best = None; start_ep = 0
    ckpt_path = args.out + ".ckpt"
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, best5, test_at_best, start_ep = ck["best"], ck["best5"], ck["test_at_best"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_torch"].cpu()); _np.random.set_state(ck["rng_np"]); _r.setstate(ck["rng_py"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        print(f"[resume] from {ckpt_path}: continue at ep {start_ep + 1}, best {best:.4f}", flush=True)

    for ep in range(start_ep, args.epochs):
        model.train(); model.clip.eval()    # CLIP stays frozen/eval
        tot = n = 0
        for step, (frames, points, label) in enumerate(tl):
            frames, points, label = frames.to(device), points.to(device), label.to(device)
            loss = lossf(model(frames, points, args.branch), label)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * label.numel(); n += label.numel()
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f"  ep {ep + 1} step {step + 1}/{len(tl)} running-loss {tot / max(n, 1):.4f} tau {model.logit_scale.item():.3f}", flush=True)
        sched.step()
        acc, acc5 = evaluate(model, vl, device, args.branch, topk=True)
        if acc >= best:
            best, best5 = acc, acc5
            if tl_test is not None:
                test_at_best = evaluate(model, tl_test, device, args.branch, topk=True)
            torch.save({"model": trainable_state(), "acc": acc, "epoch": ep, "prompts": prompts}, args.out)
            if args.dump_preds:
                dump_orders(model, vl, device, args.branch, args.dump_preds, ep)
        print(f"ep {ep + 1:3d}/{args.epochs}  loss {tot / max(n, 1):.4f}  val-top1 {acc:.4f}  best {best:.4f}  top5 {acc5:.4f}"
              + (f"  | test@best top1 {test_at_best[0]:.4f} top5 {test_at_best[1]:.4f}" if test_at_best else ""), flush=True)
        if args.resume:      # written atomically so a kill mid-save leaves the previous epoch's ckpt intact
            torch.save({"model": trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "best": best, "best5": best5, "test_at_best": test_at_best,
                        "rng_torch": torch.get_rng_state(), "rng_np": _np.random.get_state(), "rng_py": _r.getstate(),
                        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    print(f"\n[done] best val top-1 = {best:.4f}   (ExACT reports 94.83% on this split)")
    if args.resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
