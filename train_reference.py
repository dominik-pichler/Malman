"""
train_reference.py -- replicate the EMBER2024 authors' benchmark LightGBM recipe.

Fixes the three things we were doing wrong vs. the reference (examples/lgbm_config.json +
train_model in thrember/model.py):
  1. REGULARIZED + SMALL: 500 iters, 64 leaves, lambda_l2=1.0  (we had 3000 iters, 200
     leaves, no L2 -> massively overfit -> collapsed off-distribution to near-0).
  2. ALL WEEKS: random stratified 90/10 split for early stopping, trains on every week
     (we held out weeks 44-51 -> never trained on data closest to the eval).
  3. ALL FORMATS: one model over PE + non-PE (authors show APK/ELF/PDF are learnable from
     byte/string/general features) -> so we DON'T floor non-PE anymore.

Note: the reference declares dims [2,3,4,5,6,701,702] categorical. We keep them NUMERICAL
so the pure-numpy inference evaluator (which only does numerical splits) stays exact -- a
tiny deviation from the recipe, worth it for guaranteed train/inference parity.

Usage:
    uv run train_reference.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz \
                              --target-rows 2000000 --out cinder_lgbm.txt
"""
import argparse, glob, gzip, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ember_features import PEFeatureExtractor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# from examples/lgbm_config.json (categorical_feature dropped -- see module docstring)
REF_PARAMS = dict(objective="binary", boosting="gbdt", num_iterations=500, learning_rate=0.1,
                  max_depth=-1, num_leaves=64, min_data_in_leaf=100, min_sum_hessian_in_leaf=1e-3,
                  bagging_fraction=0.9, bagging_freq=1, bagging_seed=0, feature_fraction=0.9,
                  feature_fraction_bynode=0.9, feature_fraction_seed=0, lambda_l1=0.0, lambda_l2=1.0,
                  is_unbalance=True, boost_from_average=True, sigmoid=1.0, seed=0, num_threads=0,
                  verbosity=-1, metric="auc")

def load_all(cache, target_rows):
    """All formats, all weeks, subsampled stratified-ish per shard."""
    files = sorted(glob.glob(str(Path(cache) / "*.npz")))
    total = 0
    for f in files:
        total += int(np.isin(np.load(f, allow_pickle=True)["label"].astype(int), (0, 1)).sum())
    frac = min(1.0, target_rows / total) if total else 1.0
    rng = np.random.default_rng(0)
    Xs, ys, fts = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        y = d["label"].astype(int); idx = np.where(np.isin(y, (0, 1)))[0]
        if frac < 1.0:
            idx = np.sort(rng.choice(idx, max(1, int(len(idx) * frac)), replace=False))
        Xs.append(d["X"][idx]); ys.append(y[idx]); fts.append(d["file_type"][idx].astype(str))
    X = np.vstack(Xs); y = np.concatenate(ys); ft = np.concatenate(fts)
    print(f"loaded {len(X):,} rows (frac={frac:.3f} of {total:,}), all formats/weeks")
    return X, y, ft

def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

def tpr_at(y, s, t):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(t, fpr, tpr))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--shard", required=True)
    ap.add_argument("--target-rows", type=int, default=2000000); ap.add_argument("--out", default="cinder_lgbm.txt")
    args = ap.parse_args()

    X, y, ft = load_all(args.cache, args.target_rows)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    print(f"random split: train {len(ytr):,}, val {len(yval):,}  (prevalence {y.mean():.3f})")

    t0 = time.time()
    model = lgb.train(REF_PARAMS, lgb.Dataset(Xtr, ytr), valid_sets=[lgb.Dataset(Xval, yval)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {model.num_trees()} trees in {time.time()-t0:.0f}s")

    sv = model.predict(Xval)
    print(f"\n[random-split val]  PR-AUC={average_precision_score(yval, sv):.4f}  "
          f"ROC={roc_auc_score(yval, sv):.4f}  TPR@1e-3={tpr_at(yval, sv, 1e-3):.3f}")
    # eval distribution: THIS model, all formats
    ex = PEFeatureExtractor()
    recs = list(read_jsonl(args.shard))
    Xe = np.vstack([_safe(ex, r) for r in recs]).astype(np.float32)
    se = model.predict(Xe)
    fte = np.array([str(r.get("file_type")) for r in recs])
    print(f"\n=== EVAL score distribution (n={len(recs)}) -- looking for SPREAD, not near-0 ===")
    for f in sorted(set(fte)):
        v = se[fte == f]
        print(f"  {f:8} n={len(v):>6}  mean={v.mean():.3f}  p10={np.percentile(v,10):.3f}  "
              f"p50={np.percentile(v,50):.3f}  p90={np.percentile(v,90):.3f}  frac>0.5={np.mean(v>0.5):.3f}")

    model.save_model(args.out)
    print(f"\nsaved -> {args.out} ({Path(args.out).stat().st_size/1e6:.0f} MB, {model.num_trees()} trees)")
    print("NOTE: set NONPE_SCORE=None in solution.py (this model scores non-PE itself); "
          "remove cinder_drop_groups.txt if present.")

def _safe(ex, r):
    try: return np.asarray(ex.process_raw_features(r), dtype=np.float32)
    except Exception: return np.zeros(ex.dim, dtype=np.float32)

if __name__ == "__main__":
    main()