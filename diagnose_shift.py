"""
diagnose_shift.py -- why does the model score 0.998 on validation but ~0.05 (ROC<0.5) on eval?

Distinguishes the two theories:
  (A) EXTRACTOR MISMATCH: the vendored ember_features produces different vectors than the
      installed thrember the model trained on. -> parity max|diff| > 0 = BUG, and fixable.
  (B) DISTRIBUTION SHIFT: extractors agree, but eval vectors live elsewhere in feature space
      than training vectors (evasive/OOD). -> per-group shift + which groups are dead on eval.

Also checks raw-field richness: are header/authenticode/imports populated in eval records,
or empty (which would kill the model's top features)?

Usage:
    uv run diagnose_shift.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz \
                             --train-shard ../data/train/win32-shard-0000.jsonl.gz --model cinder_lgbm.txt
"""
import argparse, glob, gzip, json, sys, types
from pathlib import Path
import numpy as np

# stub signify so the INSTALLED thrember imports (same trick cinder_ember.py uses)
_a = types.ModuleType("signify.authenticode"); _a.SignedPEFile = object
sys.modules.setdefault("signify", types.ModuleType("signify"))
sys.modules.setdefault("signify.authenticode", _a)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ember_features import PEFeatureExtractor as Vendored
PE = {"Win32", "Win64", "Dot_Net"}

def read_jsonl(path, n=None):
    out = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if n and i >= n: break
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def group_ranges(ex):
    r, i = {}, 0
    for fe in ex.features:
        r[fe.name] = (i, i + fe.dim); i += fe.dim
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--shard", required=True)
    ap.add_argument("--train-shard", required=True); ap.add_argument("--model", default="cinder_lgbm.txt")
    args = ap.parse_args()

    ev = Vendored(); groups = group_ranges(ev)
    eval_recs = read_jsonl(args.shard, 1500)
    eval_pe = [r for r in eval_recs if str(r.get("file_type")) in PE]
    train_recs = read_jsonl(args.train_shard, 1500)

    # ---- (A) extractor parity: vendored vs INSTALLED thrember on the SAME eval records ----
    print("=== (A) EXTRACTOR PARITY: vendored ember_features vs installed thrember ===")
    try:
        from thrember.features import PEFeatureExtractor as Installed
        it = Installed()
        maxd = 0.0
        for r in eval_pe[:500]:
            v1 = np.asarray(ev.process_raw_features(r), dtype=np.float64)
            v2 = np.asarray(it.process_raw_features(r), dtype=np.float64)
            maxd = max(maxd, float(np.nanmax(np.abs(v1 - v2))))
        verdict = "IDENTICAL (extractor is faithful -> theory B)" if maxd < 1e-6 else \
                  f"DIFFER by {maxd:.3e} -> THIS IS THE BUG (theory A). The submitted vectors are wrong."
        print(f"  max|diff| over 500 eval records: {maxd:.3e}  -> {verdict}")
    except Exception as e:
        print(f"  installed thrember not importable ({e}); skipping A -- run where thrember is installed")

    # ---- (B) distribution shift: train cache vectors vs eval vectors, per group ----
    print("\n=== (B) FEATURE SHIFT: training vectors (cache) vs eval vectors, per group ===")
    # sample training PE vectors from cache
    Xtr = []
    for f in sorted(glob.glob(str(Path(args.cache) / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        ft = d["file_type"].astype(str); m = np.isin(ft, list(PE))
        if m.any():
            idx = np.where(m)[0][:400]; Xtr.append(d["X"][idx])
        if sum(len(a) for a in Xtr) > 8000: break
    Xtr = np.vstack(Xtr)
    Xev = np.vstack([np.asarray(ev.process_raw_features(r), dtype=np.float32) for r in eval_pe[:2000]])
    print(f"  train sample {Xtr.shape}, eval sample {Xev.shape}")
    print(f"  {'group':16} {'train_mean':>12} {'eval_mean':>12} {'train_nz%':>9} {'eval_nz%':>9}")
    rows = []
    for g, (a, b) in groups.items():
        tm, em = np.nanmean(np.abs(Xtr[:, a:b])), np.nanmean(np.abs(Xev[:, a:b]))
        tnz, enz = np.mean(Xtr[:, a:b] != 0) * 100, np.mean(Xev[:, a:b] != 0) * 100
        rows.append((abs(tm - em), g, tm, em, tnz, enz))
    for _, g, tm, em, tnz, enz in sorted(rows, reverse=True):
        flag = "  <-- DEAD on eval" if (tnz > 20 and enz < 2) else ""
        print(f"  {g:16} {tm:12.3g} {em:12.3g} {tnz:8.1f}% {enz:8.1f}%{flag}")

    # ---- (C) raw-field richness: eval vs train ----
    print("\n=== (C) RAW-FIELD RICHNESS (are the model's top features populated in eval?) ===")
    def richness(recs, tag):
        auth = np.mean([bool(r.get("authenticode", {}).get("num_certs", 0)) for r in recs]) * 100
        hdr = np.mean([bool(r.get("header", {}).get("coff", {}).get("timestamp", 0)) for r in recs]) * 100
        imp = np.mean([len(r.get("imports", {}) or {}) > 0 for r in recs]) * 100
        exp = np.mean([len(r.get("exports", []) or []) > 0 for r in recs]) * 100
        print(f"  {tag:14} has_cert={auth:5.1f}%  has_hdr_ts={hdr:5.1f}%  has_imports={imp:5.1f}%  has_exports={exp:5.1f}%")
    richness(train_recs, "train (win32)")
    richness(eval_pe, "eval (PE)")
    print("\nRead: (A) diff>0 -> fix the extractor. Else (B)/(C): if header/authenticode/imports are"
          "\n'dead on eval' or unpopulated, the eval was extracted differently; if they're populated but"
          "\nvectors still shift, the eval malware is genuinely evasive and defeats the metadata features.")

if __name__ == "__main__":
    main()