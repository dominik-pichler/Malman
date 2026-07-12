"""
train_and_probe.py -- train and score in ONE process, to remove all file-staleness
ambiguity. Prints, from a single freshly-trained model:
  * per-format PR-AUC on the temporal validation split  (can the model learn each format?)
  * per-format SCORE DISTRIBUTION on the eval shard      (does the fresh model behave differently?)

If the eval PDF/APK distributions here differ from the stuck ones, the earlier problem
was staleness -> this model is good, submit it. If they're STILL stuck despite strong
per-format validation PR-AUC, the eval's non-PE files are out-of-distribution vs train
(a real finding). The model is capped small so it won't choke the grader.

Usage:
    uv run train_and_probe.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz \
                              --target-rows 700000 --out cinder_lgbm.txt
"""
import argparse, glob, gzip, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ember_features import PEFeatureExtractor
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

VALID_WEEKS = 8

def tpr_at(y, s, t):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(t, fpr, tpr))

def load_subsampled(cache, target_rows):
    files = sorted(glob.glob(str(Path(cache) / "*.npz")))
    total = 0
    for f in files:
        total += int(np.isin(np.load(f, allow_pickle=True)["label"].astype(int), (0, 1)).sum())
    frac = min(1.0, target_rows / total) if total else 1.0
    rng = np.random.default_rng(0)
    Xs, ys, ws, fs = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        y = d["label"].astype(int); idx = np.where(np.isin(y, (0, 1)))[0]
        if frac < 1.0:
            idx = np.sort(rng.choice(idx, max(1, int(len(idx) * frac)), replace=False))
        Xs.append(d["X"][idx]); ys.append(y[idx])
        ws.append(d["week_id"][idx].astype(int)); fs.append(d["file_type"][idx].astype(str))
    print(f"loaded {sum(len(a) for a in ys):,} rows (frac={frac:.3f} of {total:,})")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(fs)

def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--shard", required=True)
    ap.add_argument("--target-rows", type=int, default=700000); ap.add_argument("--out", default="cinder_lgbm.txt")
    ap.add_argument("--formats", default="", help="comma list to keep, e.g. Win32,Win64,Dot_Net (default: all)")
    ap.add_argument("--rounds", type=int, default=2500); ap.add_argument("--leaves", type=int, default=200)
    args = ap.parse_args()

    X, y, wk, ft = load_subsampled(args.cache, args.target_rows)
    if args.formats:
        keep_ft = set(args.formats.split(","))
        m = np.isin(ft, list(keep_ft))
        X, y, wk, ft = X[m], y[m], wk[m], ft[m]
        print(f"format filter {sorted(keep_ft)}: kept {m.sum():,} rows")
    uniq = np.unique(wk[wk >= 0]); cut = uniq[-VALID_WEEKS]
    tr, va = wk < cut, wk >= cut
    print(f"temporal split: train weeks {uniq[0]}..{cut-1} (n={tr.sum():,}), "
          f"valid weeks {cut}..{uniq[-1]} (n={va.sum():,})")

    def ap_eval(p, d): return "PR_AUC", average_precision_score(d.get_label(), p), True
    params = dict(objective="binary", metric="None", learning_rate=0.05, num_leaves=args.leaves,
                  min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  max_bin=255, seed=13, verbosity=-1, n_jobs=-1)
    t0 = time.time()
    model = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=args.rounds,
                      valid_sets=[lgb.Dataset(X[va], y[va])], feval=ap_eval,
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
    print(f"trained {model.num_trees()} trees in {time.time()-t0:.0f}s")

    sv = model.predict(X[va])
    print(f"\n=== VALIDATION per-format (can the model learn each format in-distribution?) ===")
    print(f"{'ALL':8} n={va.sum():>7}  PR-AUC={average_precision_score(y[va], sv):.4f}  ROC={roc_auc_score(y[va], sv):.4f}")
    for t in sorted(set(ft[va])):
        m = ft[va] == t
        if len(set(y[va][m])) > 1:
            print(f"{t:8} n={m.sum():>7}  PR-AUC={average_precision_score(y[va][m], sv[m]):.4f}  "
                  f"ROC={roc_auc_score(y[va][m], sv[m]):.4f}  TPR@1e-3={tpr_at(y[va][m], sv[m], 1e-3):.3f}")

    # ---- score the eval shard with THIS model, same process ----
    ex = PEFeatureExtractor()
    recs = list(read_jsonl(args.shard))
    Xe = np.vstack([_safe(ex, r) for r in recs]).astype(np.float32)
    fte = np.array([str(r.get("file_type")) for r in recs])
    se = model.predict(Xe)
    print(f"\n=== EVAL score distribution from THIS fresh model (n={len(recs)}) ===")
    for t in sorted(set(fte)):
        v = se[fte == t]
        print(f"{t:8} n={len(v):>6}  mean={v.mean():.3f}  p10={np.percentile(v,10):.3f}  "
              f"p50={np.percentile(v,50):.3f}  p90={np.percentile(v,90):.3f}")

    model.save_model(args.out)
    print(f"\nsaved -> {args.out}  ({Path(args.out).stat().st_size/1e6:.0f} MB, {model.num_trees()} trees)")

def _safe(ex, r):
    try: return np.asarray(ex.process_raw_features(r), dtype=np.float32)
    except Exception: return np.zeros(ex.dim, dtype=np.float32)

if __name__ == "__main__":
    main()