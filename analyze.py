"""
analyze.py -- deep diagnostics for a trained Cinder model, on your labeled cache.

Runs five analyses on the model's temporal-validation holdout (latest weeks of the
cache, which the model was NOT trained on):

  1. OPERATING POINTS  -- PR-AUC/ROC + TPR & precision at FPR 1e-2/1e-3/1e-4, overall and per format
  2. FEATURE IMPORTANCE-- LightGBM gain aggregated by EMBER feature GROUP + top individual dims
  3. ERROR INSPECTION  -- which feature groups separate false positives from true negatives
                          (and FNs from TPs); flagged samples saved to analyze_errors.csv
  4. LEAKAGE AUDIT     -- exact (sha256) and near-duplicate (identical tlsh) overlap between
                          the train-weeks and valid-weeks -- quantifies optimistic-CV risk (H4)
  5. TEMPORAL DRIFT    -- PR-AUC per validation week

Uses lightgbm locally (fine -- lightgbm is only absent on the grader).

Usage:
    uv run analyze.py --cache cache/train --model cinder_lgbm.txt \
                      --formats Win32,Win64,Dot_Net --target-rows 600000
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ember_features import PEFeatureExtractor
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_score

VALID_WEEKS = 8


def group_map():
    """dim index -> EMBER feature-group name, from the extractor's own layout."""
    ex = PEFeatureExtractor()
    names, i = np.empty(ex.dim, dtype=object), 0
    for fe in ex.features:
        names[i:i + fe.dim] = fe.name; i += fe.dim
    return names, ex.dim


def load_meta(cache):
    """cheap: sha256/tlsh/week/label/file_type for ALL rows (no X)."""
    sha, tlsh, wk, y, ft = [], [], [], [], []
    for f in sorted(glob.glob(str(Path(cache) / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        sha.append(d["sha256"]); tlsh.append(d["tlsh"]); wk.append(d["week_id"].astype(int))
        y.append(d["label"].astype(int)); ft.append(d["file_type"].astype(str))
    return (np.concatenate(sha), np.concatenate(tlsh), np.concatenate(wk),
            np.concatenate(y), np.concatenate(ft))


def load_valid_X(cache, cut, formats, target_rows):
    """load X for valid-week rows only (weeks>=cut), format-filtered, subsampled."""
    rng = np.random.default_rng(0)
    Xs, ys, wks, fts, shas = [], [], [], [], []
    keep_ft = set(formats.split(",")) if formats else None
    for f in sorted(glob.glob(str(Path(cache) / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        wk = d["week_id"].astype(int); y = d["label"].astype(int); ft = d["file_type"].astype(str)
        m = (wk >= cut) & np.isin(y, (0, 1))
        if keep_ft is not None:
            m &= np.isin(ft, list(keep_ft))
        if m.any():
            Xs.append(d["X"][m]); ys.append(y[m]); wks.append(wk[m]); fts.append(ft[m]); shas.append(d["sha256"][m])
    X = np.vstack(Xs); y = np.concatenate(ys); wk = np.concatenate(wks); ft = np.concatenate(fts); sha = np.concatenate(shas)
    if target_rows and len(X) > target_rows:
        idx = np.sort(rng.choice(len(X), target_rows, replace=False))
        X, y, wk, ft, sha = X[idx], y[idx], wk[idx], ft[idx], sha[idx]
    return X, y, wk, ft, sha


def tpr_prec_at_fpr(y, s, t):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.where(fpr <= t)[0]
    i = i[-1] if len(i) else 0
    prec = precision_score(y, (s >= thr[i]).astype(int), zero_division=0)
    return float(tpr[i]), float(prec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--model", default="cinder_lgbm.txt")
    ap.add_argument("--formats", default="Win32,Win64,Dot_Net"); ap.add_argument("--target-rows", type=int, default=600000)
    ap.add_argument("--errors-out", default="analyze_errors.csv")
    args = ap.parse_args()

    gnames, dim = group_map()
    booster = lgb.Booster(model_file=args.model)
    print(f"model: {booster.num_trees()} trees, {dim} features\n")

    # ---- 4) leakage audit (from cheap meta, before loading X) ----
    sha, tlsh, wk, y_all, ft_all = load_meta(args.cache)
    uniq = np.unique(wk[wk >= 0]); cut = uniq[-VALID_WEEKS]
    tr_m, va_m = wk < cut, wk >= cut
    tr_sha, va_sha = set(sha[tr_m]), set(sha[va_m])
    exact = len(va_sha & tr_sha)
    tr_tl = set(t for t in tlsh[tr_m] if t); va_tl = [t for t in tlsh[va_m] if t]
    near = sum(1 for t in va_tl if t in tr_tl)
    print("=== 4) LEAKAGE AUDIT (train weeks vs valid weeks) ===")
    print(f"  valid rows: {va_m.sum():,} | exact sha256 also in train: {exact:,} "
          f"({100*exact/max(va_m.sum(),1):.2f}%)")
    print(f"  valid rows with a tlsh twin in train: {near:,} of {len(va_tl):,} with tlsh "
          f"({100*near/max(len(va_tl),1):.2f}%)  <- near-duplicate leakage / CV inflation\n")

    # ---- load valid X ----
    X, y, vwk, ft, vsha = load_valid_X(args.cache, cut, args.formats, args.target_rows)
    s = booster.predict(X)
    print(f"validation holdout: weeks {cut}..{uniq[-1]}  n={len(y):,}  formats={sorted(set(ft))}\n")

    # ---- 1) operating points ----
    print("=== 1) OPERATING POINTS ===")
    def line(tag, yy, ss):
        if len(set(yy)) < 2:
            print(f"  {tag:9} n={len(yy):>7}  (single-class)"); return
        row = f"  {tag:9} n={len(yy):>7}  PR-AUC={average_precision_score(yy,ss):.4f}  ROC={roc_auc_score(yy,ss):.4f}"
        for t in (1e-2, 1e-3, 1e-4):
            tpr, prec = tpr_prec_at_fpr(yy, ss, t)
            row += f"  | FPR{t:.0e}: TPR={tpr:.3f} P={prec:.3f}"
        print(row)
    line("ALL", y, s)
    for f in sorted(set(ft)):
        m = ft == f; line(f, y[m], s[m])

    # ---- 2) feature importance by group ----
    print("\n=== 2) FEATURE IMPORTANCE (gain) ===")
    gain = booster.feature_importance(importance_type="gain")
    grp = {}
    for g in np.unique(gnames):
        grp[g] = float(gain[gnames == g].sum())
    tot = sum(grp.values()) or 1.0
    for g, v in sorted(grp.items(), key=lambda kv: -kv[1]):
        print(f"  {g:16} {100*v/tot:5.1f}%")
    top = np.argsort(-gain)[:12]
    print("  top dims:", ", ".join(f"#{i}({gnames[i]})" for i in top))

    # ---- 3) error inspection ----
    print("\n=== 3) ERROR INSPECTION (validation) ===")
    order = np.argsort(-s)
    fp = order[(y[order] == 0)][:300]            # benign, highest scores
    fn = order[::-1][(y[order[::-1]] == 1)][:300]  # malware, lowest scores
    tn = np.where((y == 0) & (s < 0.1))[0]
    print(f"  false positives (benign, top scores): {len(fp)}  |  false negatives (malware, low scores): {len(fn)}")
    if len(fp) and len(tn):
        dgrp = {}
        for g in np.unique(gnames):
            cols = gnames == g
            dgrp[g] = float(np.abs(X[fp][:, cols].mean(0) - X[tn][:, cols].mean(0)).mean())
        print("  groups where FALSE POSITIVES differ most from true negatives:")
        for g, v in sorted(dgrp.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {g:16} mean|Δ|={v:.4f}")
    # save flagged samples for eyeballing
    import csv
    with open(args.errors_out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "sha256", "file_type", "week", "label", "score"])
        for i in fp: w.writerow(["FP", vsha[i], ft[i], vwk[i], y[i], f"{s[i]:.4f}"])
        for i in fn: w.writerow(["FN", vsha[i], ft[i], vwk[i], y[i], f"{s[i]:.4f}"])
    print(f"  wrote flagged samples -> {args.errors_out} (look these sha256 up in the raw shards)")

    # ---- 5) temporal drift ----
    print("\n=== 5) TEMPORAL DRIFT (PR-AUC per valid week) ===")
    for w in sorted(set(vwk)):
        m = vwk == w
        if len(set(y[m])) > 1:
            print(f"  week {w}: n={m.sum():>6}  PR-AUC={average_precision_score(y[m], s[m]):.4f}  "
                  f"TPR@1e-3={tpr_prec_at_fpr(y[m], s[m], 1e-3)[0]:.3f}")


if __name__ == "__main__":
    main()