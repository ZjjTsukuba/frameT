# -*- coding: utf-8 -*-
"""
DailyDVS-200 (200 action classes; AEDAT4 / DVXplorer Lite 320x240) -> compact per-clip event npz,
SAME format as PAF/SeAct (x,y,t,p,label) so the dataset/model/train pipeline is reused unchanged.

Official repo: https://github.com/QiWang233/DailyDVS-200 (ECCV'24).
Official train/val/test lists ("<relative_path>.aedat4 <label_id>" per line, label 0..199) are kept
alongside this script as DailyDVS200_{train,val,test}_official.txt; files are matched by BASENAME so the
folder layout of the archive does not matter (the zip uses DailyDvs-200/action_001/...).

Input: either the downloaded archive (--zip, read entry by entry, no full extraction needed; the zip's
CRC is verified while reading) or an extracted directory (--src). Runs in parallel and is resumable
(meta.jsonl records finished clips). Output: <out>/<idx>.npz + manifest.json (with label_map).

Derived split lists for build_splits(mode="exact"): DailyDVS200_{train,val,test}.txt (path-only).
"""
import os, json, argparse, glob, time, zipfile, tempfile, traceback
import numpy as np

W, H = 320, 240  # DVXplorer Lite
_ZF = None       # per-worker zip handle


def parse_aedat4(path):
    import dv_processing as dv
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


def read_official_split(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            relpath, label = line.rsplit(" ", 1)
            out[os.path.basename(relpath)] = int(label)
    return out


def _init_worker(zip_path):
    global _ZF
    _ZF = zipfile.ZipFile(zip_path) if zip_path else None


def process_one(job):
    """job = (k, name, ref, label, split, out_dir, cap, seed, tmpdir). ref = zip entry name or file path."""
    k, name, ref, label, split, out_dir, cap, seed, tmpdir = job
    try:
        if _ZF is not None:
            with tempfile.TemporaryDirectory(dir=tmpdir) as d:
                p = os.path.join(d, "x.aedat4")
                with _ZF.open(ref) as src, open(p, "wb") as dst:
                    while True:
                        b = src.read(1 << 22)
                        if not b:
                            break
                        dst.write(b)
                r = parse_aedat4(p)
        else:
            r = parse_aedat4(ref)
        if r is None:
            return dict(idx=k, file=name, status="empty")
        x, y, t, p_ = r[0], r[1], r[2], r[3]
        n0 = len(t)
        if cap and n0 > cap:
            idx = np.sort(np.random.default_rng([seed, k]).choice(n0, cap, replace=False))
            x, y, t, p_ = x[idx], y[idx], t[idx], p_[idx]
        outp = os.path.join(out_dir, f"{k:05d}.npz")
        np.savez_compressed(outp + ".tmp.npz", x=x.astype(np.uint16), y=y.astype(np.uint16),
                            t=t.astype(np.uint32), p=p_.astype(np.uint8), label=np.int64(label))
        os.replace(outp + ".tmp.npz", outp)
        return dict(idx=k, npz=os.path.basename(outp), file=name, split=split, label=label,
                    n_events=int(len(t)), n_raw=int(n0), dur_us=int(t.max()), status="ok")
    except Exception as e:
        return dict(idx=k, file=name, status="error", err=repr(e), tb=traceback.format_exc()[-400:])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser("DailyDVS-200 AEDAT4 -> per-clip event npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", help="downloaded DailyDvs-200.zip (read entry by entry)")
    g.add_argument("--src", help="extracted directory (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--official-splits", default=here)
    ap.add_argument("--cap", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tmpdir", default="/tmp")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N clips (test run)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    label_of, split_of = {}, {}
    for split in ["train", "val", "test"]:
        fn = os.path.join(args.official_splits, f"DailyDVS200_{split}_official.txt")
        if not os.path.exists(fn):
            print(f"WARN: missing {fn}"); continue
        for b, lab in read_official_split(fn).items():
            label_of[b] = lab; split_of[b] = split
    print(f"official split entries: {len(label_of)}", flush=True)
    if not label_of:
        raise SystemExit("no official split files found")

    if args.zip:
        z = zipfile.ZipFile(args.zip)
        entries = [(os.path.basename(i.filename), i.filename) for i in z.infolist()
                   if not i.is_dir() and i.filename.lower().endswith(".aedat4")]
    else:
        entries = [(os.path.basename(f), f) for f in glob.glob(os.path.join(args.src, "**", "*.aedat4"), recursive=True)]
    entries.sort()
    print(f"found {len(entries)} aedat4 entries", flush=True)
    entries = [e for e in entries if e[0] in label_of]
    print(f"with official label: {len(entries)}", flush=True)
    if args.limit:
        entries = entries[:args.limit]

    meta_path = os.path.join(args.out, "meta.jsonl")
    done = {}
    if os.path.exists(meta_path):
        for line in open(meta_path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("status") == "ok" and os.path.exists(os.path.join(args.out, d["npz"])):
                done[d["idx"]] = d
    print(f"already done (resume): {len(done)}", flush=True)

    jobs = [(k, n, ref, label_of[n], split_of[n], args.out, args.cap, args.seed, args.tmpdir)
            for k, (n, ref) in enumerate(entries) if k not in done]
    print(f"to process: {len(jobs)} with {args.workers} workers", flush=True)

    from multiprocessing import Pool
    t0 = time.time(); n_ok = n_bad = 0
    with open(meta_path, "a") as mf, open(os.path.join(args.out, "errors.log"), "a") as ef:
        with Pool(args.workers, initializer=_init_worker, initargs=(args.zip,)) as pool:
            for i, res in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4), 1):
                if res["status"] == "ok":
                    mf.write(json.dumps(res, ensure_ascii=False) + "\n"); mf.flush()
                    done[res["idx"]] = res; n_ok += 1
                else:
                    ef.write(json.dumps(res, ensure_ascii=False) + "\n"); ef.flush(); n_bad += 1
                if i % 200 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  [{i}/{len(jobs)}] ok={n_ok} bad={n_bad} elapsed={el/60:.1f}min "
                          f"eta={(len(jobs)-i)*el/max(i,1)/60:.1f}min", flush=True)

    items = [done[k] for k in sorted(done)]
    man = dict(W=W, H=H, cap=args.cap, num_classes=200, items=items)
    classes_json = os.path.join(here, "DailyDVS200_classes.json")
    if os.path.exists(classes_json):
        names = json.load(open(classes_json, encoding="utf-8"))
        man["label_map"] = {name: i for i, name in enumerate(names)}
    for it in man["items"]:
        it.pop("status", None)
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), ensure_ascii=False)
    print(f"done: {len(items)} clips in manifest (bad this run: {n_bad}) -> {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
