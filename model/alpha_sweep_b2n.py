# -*- coding: utf-8 -*-
"""
Late-fusion alpha sweep for base-to-new: combine INDEPENDENTLY-trained frame->text and
point->text heads as  score = a*cos(frame,text) + (1-a)*cos(point,text),  argmax over the
class subset. Tells us the CEILING of motion's complementary value on NEW (unseen) classes:
if some alpha gives combined-new > frame-only-new (a=1), motion carries complementary signal
that appearance misses. Oracle-alpha (best on the eval set) = upper bound diagnostic. New file.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paf.paf_dataset import build_splits, PAFEvents
from paf.model_v2 import PAFClipPointV2
from paf.train_paf_b2n import split_base_new


def load_into(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect(model_f, model_p, loader, device, cols, to_local):
    """Return cos_f [n,|cols|], cos_p [n,|cols|], labels-local [n] over a subset of classes."""
    CF, CP, LB = [], [], []
    for frames, points, label in loader:
        frames, points = frames.to(device), points.to(device)
        tf = model_f._text_feats().t()[:, cols]                       # [dim, |cols|] (frozen, same both)
        f = F.normalize(model_f._img_tokens(frames).mean(1), dim=-1)  # appearance
        p = F.normalize(model_p.point(points), dim=-1)                # motion
        CF.append((f @ tf).cpu()); CP.append((p @ tf).cpu()); LB.append(to_local[label])
    return torch.cat(CF), torch.cat(CP), torch.cat(LB)


def acc_at(cf, cp, lb, a):
    return ((a * cf + (1 - a) * cp).argmax(1) == lb).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/datasets/SeAct/preprocessed")
    ap.add_argument("--classes", default="paf/SeAct_classes.json")
    ap.add_argument("--train-file", default="SeAct_train_norm.txt")
    ap.add_argument("--val-file", default="SeAct_val_norm.txt")
    ap.add_argument("--logdir", default="paf/b2n_logs")
    ap.add_argument("--splits", default="0,1,2")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    names = json.load(open(args.classes))
    if isinstance(names, dict):
        names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
    classnames = [n.split(":")[0].replace("-", " ").strip() for n in names]
    prompts = [f"a photo of a person {n}" for n in classnames]

    alphas = np.linspace(0, 1, 21)
    agg = {"new": {"frame": [], "point": [], "oracle": [], "a0.7": [], "bestA": []},
           "base": {"frame": [], "oracle_at_newA": []}}
    for S in [int(x) for x in args.splits.split(",")]:
        base_ids, new_ids = split_base_new(classnames, 0.5, "random", S)
        bset, nset = set(base_ids), set(new_ids)
        new_cols = torch.tensor(new_ids, device=dev)
        base_cols = torch.tensor(base_ids, device=dev)
        to_new = torch.full((len(classnames),), -1, dtype=torch.long)
        to_base = torch.full((len(classnames),), -1, dtype=torch.long)
        for j, c in enumerate(new_ids): to_new[c] = j
        for j, c in enumerate(base_ids): to_base[c] = j

        tr, te, man = build_splits(os.path.join(args.root, "manifest.json"), mode="exact",
                                   train_file=args.train_file, val_file=args.val_file)
        items = man["items"]
        te_new = [i for i in te if items[i]["label"] in nset]
        te_base = [i for i in te if items[i]["label"] in bset]

        def mk(idx):
            ds = PAFEvents(args.root, idx, man, T=args.T, num_points=args.points, train=False,
                           denoise=True, frame_repr="hist")
            return DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        model_f = load_into(PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder="pointmamba",
                            fusion="gated", text="hand", lora_r=4).to(dev), f"{args.logdir}/frameW_s{S}.pth")
        model_p = load_into(PAFClipPointV2(classnames, hand_prompts=prompts, point_encoder="pointmamba",
                            fusion="gated", text="hand", lora_r=0).to(dev), f"{args.logdir}/pointW_s{S}.pth")

        cf, cp, lb = collect(model_f, model_p, mk(te_new), dev, new_cols, to_new)
        accs = [acc_at(cf, cp, lb, a) for a in alphas]
        i_best = int(np.argmax(accs))
        fr, pt, orc, a07 = accs[-1], accs[0], accs[i_best], acc_at(cf, cp, lb, 0.7)
        # base accuracy at the same oracle-alpha (to report H honestly)
        cfb, cpb, lbb = collect(model_f, model_p, mk(te_base), dev, base_cols, to_base)
        base_at = acc_at(cfb, cpb, lbb, alphas[i_best])
        print(f"s{S}: NEW frame(a=1) {fr:.1f} | point(a=0) {pt:.1f} | a=0.7 {a07:.1f} | "
              f"ORACLE {orc:.1f}@a={alphas[i_best]:.2f}  (base@thatA {base_at:.1f})")
        agg["new"]["frame"].append(fr); agg["new"]["point"].append(pt)
        agg["new"]["oracle"].append(orc); agg["new"]["a0.7"].append(a07); agg["new"]["bestA"].append(alphas[i_best])
        agg["base"]["frame"].append(acc_at(cfb, cpb, lbb, 1.0)); agg["base"]["oracle_at_newA"].append(base_at)

    import statistics as st
    def m(v): return f"{st.mean(v):.1f}±{st.pstdev(v):.1f}"
    print("\n=== NEW (unseen) mean over splits ===")
    print(f" frame-only (a=1) : {m(agg['new']['frame'])}")
    print(f" point-only (a=0) : {m(agg['new']['point'])}")
    print(f" fixed a=0.7      : {m(agg['new']['a0.7'])}")
    print(f" ORACLE-alpha     : {m(agg['new']['oracle'])}   (bestA {m(agg['new']['bestA'])})")
    print(f"  -> motion complementary gain (oracle - frame): "
          f"{st.mean(agg['new']['oracle']) - st.mean(agg['new']['frame']):+.2f}")


if __name__ == "__main__":
    main()
