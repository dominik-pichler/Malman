"""
check_submission.py -- pre-flight validation of solution.py WITHOUT eval labels.

The grader is the only score oracle now, so every submission is precious. This
script verifies everything that can be verified offline before spending one:

  1. COUNT      : one score per record for the real eval shard (the grader's format check)
  2. PARITY     : solution.py's pure-numpy scores == real LightGBM's predict() on the
                  SAME records (catches evaluator bugs like the NaN mis-routing, no labels needed)
  3. PROXY SCORE: runs solution.py on YOUR labeled temporal holdout (latest train weeks
                  from cache/train) and prints the grader's metric block on it -- the best
                  labels-free estimate of the submitted score's ballpark
  4. TIMING + score-distribution sanity per format

Usage (run in the project, where cache/train exists and lightgbm is installed):
    uv run check_submission.py --shard ../data/evaluation/shard-0000.jsonl.gz \
                               --cache cache/train
"""
import argparse, glob, gzip, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution
from solution import predict_malware


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="eval shard jsonl.gz")
    ap.add_argument("--cache", default=None, help="train cache dir for the labeled proxy score")
    ap.add_argument("--proxy-weeks", type=int, default=4, help="latest N weeks of train as proxy holdout")
    args = ap.parse_args()

    # ---- 1) count + timing on the real eval shard --------------------------------
    t0 = time.time()
    scores = predict_malware([Path(args.shard)])
    dt = time.time() - t0
    meta = [(r.get("sha256"), r.get("file_type")) for r in read_jsonl(args.shard)]
    n_rec = len(meta)
    ok_count = len(scores) == n_rec
    print(f"[count ] records={n_rec}  scores={len(scores)}  -> {'OK' if ok_count else 'MISMATCH (grader will reject)'}")
    print(f"[timing] {dt:.1f}s for the full shard")

    s = np.asarray(scores, float)
    fts = np.array([ft for _, ft in meta])
    print("[scores] per-format distribution (labels-free sanity):")
    for ft in sorted(set(fts)):
        v = s[fts == ft]
        print(f"    {ft:8} n={len(v):>6}  mean={v.mean():.3f}  p10={np.percentile(v,10):.3f}  "
              f"p50={np.percentile(v,50):.3f}  p90={np.percentile(v,90):.3f}")

    # ---- 2) parity: numpy evaluator vs real lightgbm on the SAME records ----------
    try:
        import lightgbm as lgb
        from ember_features import PEFeatureExtractor
        ex = PEFeatureExtractor()
        recs = []
        for i, r in enumerate(read_jsonl(args.shard)):
            if i >= 2000:
                break
            recs.append(r)
        X = np.vstack([_safe_vec(ex, r) for r in recs]).astype(np.float32)
        booster = lgb.Booster(model_file=str(Path(__file__).parent / "cinder_lgbm.txt"))
        p_ref = booster.predict(X)
        p_np = solution._predict(X)
        diff = float(np.max(np.abs(p_ref - p_np)))
        print(f"[parity] numpy evaluator vs lightgbm on {len(recs)} real eval records: "
              f"max|diff|={diff:.2e}  -> {'OK' if diff < 1e-6 else 'BUG: evaluator diverges'}")
    except ImportError:
        print("[parity] lightgbm not importable here -- skipped (run where you trained)")

    # ---- 3) labeled proxy score on your own temporal holdout ----------------------
    if args.cache:
        files = sorted(glob.glob(str(Path(args.cache) / "*.npz")))
        if not files:
            print(f"[proxy ] no caches in {args.cache} -- skipped")
            return
        Xs, ys, wks = [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            Xs.append(d["X"]); ys.append(d["label"].astype(int)); wks.append(d["week_id"].astype(int))
        X = np.vstack(Xs); y = np.concatenate(ys); wk = np.concatenate(wks)
        lab = np.isin(y, (0, 1)); X, y, wk = X[lab], y[lab], wk[lab]
        uniq = np.unique(wk[wk >= 0]); cut = uniq[-args.proxy_weeks]
        m = wk >= cut
        p = solution._predict(X[m])
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
        def tpr_at(yv, sv, t):
            fpr, tpr, _ = roc_curve(yv, sv); return float(np.interp(t, fpr, tpr))
        print(f"[proxy ] labeled holdout = train weeks {cut}..{uniq[-1]}  n={m.sum():,}")
        print(f"         PR-AUC={average_precision_score(y[m], p):.4f}  ROC-AUC={roc_auc_score(y[m], p):.4f}  "
              f"TPR@1%={tpr_at(y[m], p, 1e-2):.3f}  TPR@0.1%={tpr_at(y[m], p, 1e-3):.3f}")
        print("         (upper-bound estimate: eval is later in time + partly non-PE)")


def _safe_vec(ex, r):
    try:
        return np.asarray(ex.process_raw_features(r), dtype=np.float32)
    except Exception:
        return np.zeros(ex.dim, dtype=np.float32)


if __name__ == "__main__":
    main()