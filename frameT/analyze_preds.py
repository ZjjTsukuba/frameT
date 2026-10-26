# -*- coding: utf-8 -*-
"""Analysis of the queue-v9 prediction dumps (results/thesis4_preds/*.npz).
Outputs (results/thesis4_analysis/): order_test.md, per_class_<ds>.md, confusion_<ds>_<cfg>.png
 - frame-order test: val top-1 with original / reversed / shuffled frame order (mean ± std over seeds)
 - per-class accuracy, proposed (hist, Mamba) vs mean-pool (noTemp), pooled over seeds
 - most-confused class pairs + row-normalized confusion matrix (summed over seeds)
Usage: python analyze_preds.py"""
import os, json, glob
import numpy as np

PRED = os.path.expanduser("~/research/results/thesis4_preds")
OUT = os.path.expanduser("~/research/results/thesis4_analysis")
CODE = os.path.expanduser("~/research/frameT/frameT")
os.makedirs(OUT, exist_ok=True)


def classnames(ds):
    if ds == "seact":
        names = json.load(open(f"{CODE}/SeAct_classes.json"))
        if isinstance(names, dict):
            names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
        return [n.split(":")[0].replace("-", " ").strip() for n in names]
    if ds == "dd":
        names = json.load(open(f"{CODE}/DailyDVS200_classes.json"))
        if isinstance(names, dict):
            names = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
        return [n.split(":")[0].replace("-", " ").strip() for n in names]
    man = json.load(open(os.path.expanduser("~/research/data/PAF/preprocessed/manifest.json")))
    inv = {v: k for k, v in man["label_map"].items()}
    return [inv[i] for i in range(len(inv))]


def load(ds, cfg):
    return [np.load(f) for f in sorted(glob.glob(f"{PRED}/{ds}_{cfg}_s*.npz"))]


def acc(d, k):
    return 100 * (d[k].argmax(1) == d["labels"]).mean()


lines = ["# フレーム順序テスト（val top-1，best epoch，seed 平均 ± 標準偏差）", "",
         "| データセット | 設定 | n seed | 元の順序 | 逆順 | シャッフル |", "|---|---|---|---|---|---|"]
for ds in ["seact", "paf", "dd"]:
    for cfg in ["hist", "noTemp", "histt"]:
        runs = load(ds, cfg)
        if not runs:
            continue
        cells = []
        for k in ["orig", "rev", "shuf"]:
            a = np.array([acc(d, k) for d in runs])
            cells.append(f"{a.mean():.2f} ± {a.std(ddof=1) if len(a) > 1 else 0:.2f}")
        lines.append(f"| {ds} | {cfg} | {len(runs)} | " + " | ".join(cells) + " |")
open(f"{OUT}/order_test.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

for ds in ["seact", "paf"]:
    A, B = load(ds, "hist"), load(ds, "noTemp")
    if not A or not B:
        continue
    names = classnames(ds)
    C = len(names)

    def per_class(runs):
        cor = np.zeros(C); tot = np.zeros(C); cm = np.zeros((C, C))
        for d in runs:
            p = d["orig"].argmax(1)
            for y, q in zip(d["labels"], p):
                tot[y] += 1; cor[y] += (y == q); cm[y, q] += 1
        return 100 * cor / np.maximum(tot, 1), tot, cm

    pa, tot, cma = per_class(A)
    pb, _, cmb = per_class(B)
    diff = pa - pb
    order = np.argsort(-diff)
    out = [f"# クラス別精度：提案（Mamba）vs 時間平均（{ds}，{len(A)} seed 合算）", "",
           "| クラス | 評価数（合算） | 時間平均 | 提案 | 差 |", "|---|---|---|---|---|"]
    for i in order:
        if tot[i] == 0:
            continue
        out.append(f"| {names[i]} | {int(tot[i])} | {pb[i]:.1f} | {pa[i]:.1f} | {diff[i]:+.1f} |")

    def confused(cm, k=10):
        m = cm.copy(); np.fill_diagonal(m, 0)
        idx = np.dstack(np.unravel_index(np.argsort(-m.ravel()), m.shape))[0][:k]
        return [f"| {names[t]} | {names[p]} | {int(m[t, p])} |" for t, p in idx if m[t, p] > 0]

    out += ["", "## 混同の多いペア（正解 → 予測，件数）時間平均", "", "| 正解 | 予測 | 件数 |", "|---|---|---|"] + confused(cmb)
    out += ["", "## 混同の多いペア 提案", "", "| 正解 | 予測 | 件数 |", "|---|---|---|"] + confused(cma)
    open(f"{OUT}/per_class_{ds}.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"[{ds}] improved {np.sum(diff > 0)} / same {np.sum(diff == 0)} / worse {np.sum(diff < 0)} classes")

    if plt is not None:
        for cfg, cm in [("hist", cma), ("noTemp", cmb)]:
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            s = 14 if C > 20 else 6
            fig, ax = plt.subplots(figsize=(s, s * 0.9))
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(C)); ax.set_yticks(range(C))
            fs = 5 if C > 20 else 8
            ax.set_xticklabels(names, rotation=90, fontsize=fs); ax.set_yticklabels(names, fontsize=fs)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
            fig.colorbar(im, fraction=0.03)
            fig.tight_layout(); fig.savefig(f"{OUT}/confusion_{ds}_{cfg}.png", dpi=200); plt.close(fig)
print("written to", OUT)
