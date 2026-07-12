"""
winnable.py -- is the eval actually winnable, or genuinely novel/evasive?

Answers "why is the score so bad" by testing whether the model catches eval malware
that is a NEAR-TWIN (identical TLSH) of malware it trained on:

  * eval PE with a TRAIN-MALWARE twin, scored HIGH  -> model works on catchable malware;
    the low eval score is the truly novel/evasive tail = a real ceiling.
  * those twins scored LOW  -> the model fails even on recognizable malware = fixable
    distribution/feature problem, worth more effort.

TLSH is a fuzzy hash: an identical TLSH means near-identical files. Exact-match is a
cheap, definitive lower bound on "how much of the eval is a known-malware variant."

Applies the same feature-group drops as deployment (reads cinder_drop_groups.txt).

Usage:
    uv run winnable.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz --model cinder_lgbm.txt
"""
import argparse, glob, gzip, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ember_features import PEFeatureExtractor
import lightgbm as lgb

PE = {"Win32", "Win64", "Dot_Net"}

def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def group_ranges(ex):
    r, i = {}, 0
    for fe in ex.features:
        r[fe.name] = (i, i + fe.dim); i += fe.dim
    return r

def dist(v, tag):
    v = np.asarray(v)
    if not len(v):
        print(f"  {tag:34} (none)"); return
    print(f"  {tag:34} n={len(v):>6}  mean={v.mean():.3f}  p50={np.percentile(v,50):.3f}  "
          f"p90={np.percentile(v,90):.3f}  frac>0.5={np.mean(v>0.5):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--shard", required=True)
    ap.add_argument("--model", default="cinder_lgbm.txt")
    args = ap.parse_args()

    ex = PEFeatureExtractor()
    # apply the same group drops the deployed model uses
    drop_ranges = []
    dropfile = Path("cinder_drop_groups.txt")
    if dropfile.exists():
        gr = group_ranges(ex)
        dropped = [l.strip() for l in dropfile.read_text().split() if l.strip() in gr]
        drop_ranges = [gr[g] for g in dropped]
        print(f"applying dropped groups from sidecar: {dropped}")

    # ---- train tlsh -> label sets ----
    mal_tlsh, ben_tlsh = set(), set()
    for f in sorted(glob.glob(str(Path(args.cache) / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        tl = d["tlsh"].astype(str); y = d["label"].astype(int)
        for t, l in zip(tl, y):
            if not t:
                continue
            if l == 1: mal_tlsh.add(t)
            elif l == 0: ben_tlsh.add(t)
    print(f"train tlsh index: {len(mal_tlsh):,} malware, {len(ben_tlsh):,} benign\n")

    # ---- score eval PE, categorize by tlsh twin ----
    model = lgb.Booster(model_file=args.model)
    buckets = {"twin=train-malware": [], "twin=train-benign": [], "twin=both": [], "no twin (novel)": []}
    n_pe = 0
    batchX, batch_meta = [], []
    def flush():
        if not batchX: return
        X = np.vstack(batchX).astype(np.float32)
        for a, b in drop_ranges: X[:, a:b] = 0.0
        s = model.predict(X)
        for sc, cat in zip(s, batch_meta): buckets[cat].append(float(sc))
        batchX.clear(); batch_meta.clear()

    for r in read_jsonl(args.shard):
        if str(r.get("file_type")) not in PE:
            continue
        n_pe += 1
        t = str(r.get("tlsh") or "")
        inm, inb = (t in mal_tlsh), (t in ben_tlsh)
        cat = ("twin=both" if inm and inb else "twin=train-malware" if inm
               else "twin=train-benign" if inb else "no twin (novel)")
        try:
            v = np.asarray(ex.process_raw_features(r), dtype=np.float32)
        except Exception:
            v = np.zeros(ex.dim, dtype=np.float32)
        batchX.append(v); batch_meta.append(cat)
        if len(batchX) >= 4000: flush()
    flush()

    print(f"eval PE records: {n_pe:,}")
    tw = len(buckets['twin=train-malware']) + len(buckets['twin=both'])
    print(f"of these, {tw:,} have an identical-TLSH twin among TRAIN MALWARE "
          f"({100*tw/max(n_pe,1):.2f}%) -- the 'catchable' set\n")
    print("model score distribution by twin category:")
    for cat, v in buckets.items():
        dist(v, cat)

    print("\nHOW TO READ:")
    print("  * 'twin=train-malware' scored HIGH  -> model catches known variants; the eval's low")
    print("    score is genuinely novel/evasive malware = real ceiling, static features maxed out.")
    print("  * 'twin=train-malware' scored LOW   -> model misses even recognizable malware =")
    print("    fixable feature/distribution problem, worth pushing on.")
    print("  * few/zero twins at all -> the eval is almost entirely novel; expected to be hard.")

if __name__ == "__main__":
    main()