"""
Cinder / EMBER2024 malware-detection pipeline (v2).

v2 adds, on top of the vectorize/train/eval core:
  * parallel vectorization (--jobs) -- the 2.5M-row step is the slow one
  * `selftest`  : vectorize ONE real record and report, to catch any thrember/env
                  mismatch in seconds before committing to a multi-hour run
  * fail-fast   : warn if too few records vectorize in a shard (signals version drift)
  * calibration + low-FPR threshold: isotonic calibrator and the decision threshold
                  that hits a target FPR are fit on validation, saved, and applied at eval
  * tlsh stored in meta for optional near-duplicate analysis

The vectorizer (official `thrember`, feature v3 -> 2568 dims) reads only static PE
groups, so AV metadata (detection_ratio, dates, family, tags) never enters the model.

Install with uv (skips thrember's fragile signify/oscrypto stack -- signify is stubbed below):
    uv pip install lightgbm pefile numpy polars scikit-learn tqdm
    uv pip install "git+https://github.com/FutureComputing4AI/EMBER2024.git" --no-deps

Usage:
    uv run python cinder_ember.py selftest  --data ../../data/train
    uv run python cinder_ember.py vectorize --data ../../data/train --cache cache/train --jobs 8
    uv run python cinder_ember.py vectorize --data ../../data/eval  --cache cache/eval  --jobs 8 --glob "shard-*.jsonl.gz"
    uv run python cinder_ember.py train     --cache cache/train --max-rows 800000 --target-fpr 1e-3
    uv run python cinder_ember.py eval      --cache cache/eval  --labels ../../data/eval/labels.jsonl.gz

Note: --data/train holds the full multi-format EMBER2024; the Cinder challenge scope is
the Windows PE shards, so vectorize defaults to glob 'win*-shard-*.jsonl.gz' (Win32+Win64).
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, pickle, sys, types
import numpy as np

# stub signify BEFORE importing thrember (only pre-extracted dicts are vectorized; the
# native signify/oscrypto stack is never exercised). Runs in child procs too (spawn re-imports).
_sig = types.ModuleType("signify.authenticode"); _sig.SignedPEFile = object
sys.modules.setdefault("signify", types.ModuleType("signify"))
sys.modules.setdefault("signify.authenticode", _sig)

SEED, VALID_WEEKS = 13, 8
TARGET_FPRS = [1e-2, 1e-3, 1e-4]
HASH, LABEL, WEEK, TIME, ARCH, TLSH = "sha256", "label", "week_id", "first_submission_date", "file_type", "tlsh"
META_COLS = [HASH, LABEL, WEEK, TIME, ARCH, TLSH]
rng = np.random.default_rng(SEED)

from sklearn.metrics import average_precision_score, roc_curve, precision_score
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------- vectorization core
def iter_records(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def vectorize_file(path, extractor):
    X, meta = [], {c: [] for c in META_COLS}
    n_lab = n_fail = 0
    for r in iter_records(path):
        lab = r.get(LABEL, -1)
        n_lab += 1
        try:
            v = np.asarray(extractor.process_raw_features(r), dtype=np.float32)
        except Exception:
            n_fail += 1
            continue
        X.append(v)
        for c, val in ((HASH, r.get(HASH)), (LABEL, lab), (WEEK, r.get(WEEK, -1)),
                       (TIME, r.get(TIME, 0)), (ARCH, r.get(ARCH, "?")), (TLSH, r.get(TLSH, ""))):
            meta[c].append(val)
    Xa = np.vstack(X).astype(np.float32) if X else np.empty((0, 2568), np.float32)
    return Xa, {k: np.array(v) for k, v in meta.items()}, n_lab, n_fail

# --- worker plumbing for --jobs (spawn-safe: top-level fn + per-worker extractor) ---
_EX = None
def _init_worker():
    global _EX
    from thrember.features import PEFeatureExtractor
    _EX = PEFeatureExtractor()

def _shard_worker(job):
    path, out = job
    X, meta, n_lab, n_fail = vectorize_file(path, _EX)
    np.savez_compressed(out, X=X, **meta)
    return os.path.basename(out), len(X), n_lab, n_fail


def cmd_vectorize(args):
    os.makedirs(args.cache, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.data, args.glob)))
    if not paths:
        sys.exit(f"no shards matched '{args.glob}' under {args.data}\n"
                 "(this folder holds the full multi-format EMBER2024; the Cinder scope is "
                 "the Windows PE shards -- default glob 'win*-shard-*.jsonl.gz'. For the eval "
                 "folder pass e.g. --glob 'shard-*.jsonl.gz'.)")
    print(f"matched {len(paths)} shards for glob '{args.glob}'")
    jobs = []
    for p in paths:
        out = os.path.join(args.cache, os.path.basename(p).replace(".jsonl.gz", ".npz"))
        if os.path.exists(out) and not args.overwrite:
            print(f"  skip (cached) {os.path.basename(out)}"); continue
        jobs.append((p, out))
    if not jobs:
        print("nothing to do (all cached)"); return
    print(f"vectorizing {len(jobs)} shards with jobs={args.jobs} -> {args.cache}")

    def handle(name, n, n_lab, n_fail):
        warn = ""
        if n_lab and n_fail / n_lab > 0.5:
            warn = f"  !! {n_fail}/{n_lab} FAILED -- likely a thrember/data mismatch; run `selftest`"
        print(f"  {name}: {n} rows" + (f" ({n_fail} skipped)" if n_fail else "") + warn)

    if args.jobs <= 1:
        _init_worker()
        for j in jobs:
            handle(*_shard_worker(j))
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.jobs, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_shard_worker, jobs):
                handle(*res)


def cmd_selftest(args):
    """Vectorize the first real record from --data and report. Seconds, not hours."""
    _init_worker()
    matches = sorted(glob.glob(os.path.join(args.data, args.glob)))
    if not matches:
        sys.exit(f"no shards matched '{args.glob}' under {args.data}")
    p = matches[0]
    r = next(iter_records(p))
    ft = str(r.get(ARCH))
    print(f"shard: {os.path.basename(p)}")
    print(f"record keys: {len(r)}  | label={r.get(LABEL)}  week_id={r.get(WEEK)}  arch={ft}")
    if ft not in ("Win32", "Win64", "Dot_Net"):
        print(f"  !! WARNING: file_type '{ft}' is not a Windows PE. The PE extractor will "
              f"zero-fill PE groups on it. Narrow --glob to win*-shard-*.jsonl.gz.")
    try:
        v = np.asarray(_EX.process_raw_features(r), dtype=np.float32)
        print(f"OK: vectorized to {v.shape[0]} dims, finite={np.isfinite(v).all()}, "
              f"nonzero={(v!=0).sum()}  -> thrember matches the data, safe to run `vectorize`")
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(f"\nFAILED on a feature group -> {type(e).__name__}: {e}\n"
                 "thrember version does not match this data. Paste this and I'll pin the right version.")


# ---------------------------------------------------------------- cache loading
def load_cache(cache_dir, max_rows=None, arch=None):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not files:
        sys.exit(f"no caches in {cache_dir} -- run `vectorize` first")
    Xs, M = [], {c: [] for c in META_COLS}
    for f in files:
        d = np.load(f, allow_pickle=True)
        Xs.append(d["X"]); [M[c].append(d[c]) for c in META_COLS]
    X = np.vstack(Xs); meta = {c: np.concatenate(M[c]) for c in META_COLS}
    if arch:
        m = np.char.lower(meta[ARCH].astype(str)) == arch.lower()
        X, meta = X[m], {c: v[m] for c, v in meta.items()}
    if max_rows and len(X) > max_rows:
        idx = np.sort(rng.choice(len(X), max_rows, replace=False))
        X, meta = X[idx], {c: v[idx] for c, v in meta.items()}
    print(f"loaded {len(X):,} rows x {X.shape[1]} dims from {len(files)} shards")
    return X, meta


# --------------------------------------------------------------------- metrics
def tpr_at_fpr(y, s, f):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(f, fpr, tpr))

def threshold_at_fpr(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    i = ok[-1] if len(ok) else 0
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def report(tag, y, s):
    if len(np.unique(y)) < 2:
        print(f"[{tag}] (single-class, skipped)"); return
    line = f"[{tag}] n={len(y):>7} PR-AUC={average_precision_score(y, s):.4f}"
    for f in TARGET_FPRS:
        line += f" | TPR@{f:.0e}={tpr_at_fpr(y, s, f):.3f}"
    print(line)


# --------------------------------------------------------------------- training
def temporal_split(meta):
    weeks = meta[WEEK].astype(int)
    uniq = np.unique(weeks[weeks >= 0])
    if len(uniq) <= VALID_WEEKS:
        t = meta[TIME].astype(float); cut = np.quantile(t, 0.85)
        return t < cut, t >= cut
    cutoff = uniq[-VALID_WEEKS]
    print(f"temporal split: train weeks {uniq[0]}..{cutoff-1}, valid weeks {cutoff}..{uniq[-1]}")
    return weeks < cutoff, weeks >= cutoff

def cmd_train(args):
    import lightgbm as lgb
    X, meta = load_cache(args.cache, max_rows=args.max_rows)
    y = meta[LABEL].astype(int)
    keep = np.isin(y, (0, 1))
    if not keep.all():
        X = X[keep]; meta = {c: v[keep] for c, v in meta.items()}; y = y[keep]
        print(f"kept {keep.sum():,} labeled rows (dropped {(~keep).sum():,} unlabeled)")
    tr, va = temporal_split(meta)
    print(f"train={tr.sum():,}  valid={va.sum():,}  prevalence(valid)={y[va].mean():.3f}")

    dtr = lgb.Dataset(X[tr], y[tr]); dva = lgb.Dataset(X[va], y[va], reference=dtr)
    def ap_eval(p, d): return "PR_AUC", average_precision_score(d.get_label(), p), True
    params = dict(objective="binary", metric="None", learning_rate=0.03, num_leaves=256,
                  min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8,
                  bagging_freq=1, max_bin=255, seed=SEED, verbosity=-1, n_jobs=-1)
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva], feval=ap_eval,
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])

    s = model.predict(X[va]); yv = y[va]; print()
    report("valid", yv, s)
    wv = meta[WEEK][va].astype(int); med = int(np.median(wv))
    report(f"time:weeks<{med}", yv[wv < med], s[wv < med])
    report(f"time:weeks>={med}", yv[wv >= med], s[wv >= med])
    for a in np.unique(meta[ARCH][va]):
        m = meta[ARCH][va] == a; report(f"arch:{a}", yv[m], s[m])

    # calibration + operating point for the stated low-FPR goal
    iso = IsotonicRegression(out_of_bounds="clip").fit(s, yv)
    thr, tpr_op, fpr_op = threshold_at_fpr(yv, s, args.target_fpr)
    prec = precision_score(yv, (s >= thr).astype(int), zero_division=0)
    print(f"\noperating point @ FPR<={args.target_fpr:.0e}: threshold={thr:.5f}  "
          f"TPR={tpr_op:.3f}  precision={prec:.3f}  (realized FPR={fpr_op:.1e})")

    model.save_model(args.model)
    with open(args.model + ".op.pkl", "wb") as f:
        pickle.dump({"iso": iso, "threshold": thr, "target_fpr": args.target_fpr}, f)
    print(f"saved model -> {args.model}  (+ .op.pkl calibrator/threshold)")


# --------------------------------------------------------------------- evaluation
def cmd_eval(args):
    import lightgbm as lgb
    model = lgb.Booster(model_file=args.model)
    op = {}
    if os.path.exists(args.model + ".op.pkl"):
        op = pickle.load(open(args.model + ".op.pkl", "rb"))
    X, meta = load_cache(args.cache)
    raw = model.predict(X)
    y = meta[LABEL].astype(int)
    if args.labels and (set(np.unique(y)) - {0, 1}):
        lab = {r.get(HASH): r.get(LABEL, r.get("label")) for r in iter_records(args.labels)}
        y = np.array([lab.get(h, -1) for h in meta[HASH]], dtype=int)
    m = np.isin(y, (0, 1))
    print(f"eval rows scored: {m.sum():,} of {len(y):,}")
    if m.any():
        report("EVAL", y[m], raw[m])
        if "target_fpr" in op:
            print(f"at train operating point (target FPR {op['target_fpr']:.0e}): "
                  f"TPR={tpr_at_fpr(y[m], raw[m], op['target_fpr']):.3f}")

    prob = op["iso"].predict(raw) if "iso" in op else raw
    dec = (raw >= op["threshold"]).astype(int) if "threshold" in op else np.zeros(len(raw), int)
    rows = np.column_stack([meta[HASH], raw, prob, dec])
    np.savetxt(args.out, rows, fmt="%s", delimiter=",",
               header="sha256,score,prob_calibrated,decision", comments="")
    print(f"wrote {args.out}  <- confirm which column the challenge wants as the submission")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest"); st.add_argument("--data", required=True)
    st.add_argument("--glob", default="win*-shard-*.jsonl.gz"); st.set_defaults(func=cmd_selftest)
    v = sub.add_parser("vectorize"); v.add_argument("--data", required=True); v.add_argument("--cache", required=True)
    v.add_argument("--glob", default="win*-shard-*.jsonl.gz"); v.add_argument("--jobs", type=int, default=1)
    v.add_argument("--overwrite", action="store_true")
    v.set_defaults(func=cmd_vectorize)
    t = sub.add_parser("train"); t.add_argument("--cache", required=True); t.add_argument("--max-rows", type=int, default=None)
    t.add_argument("--target-fpr", type=float, default=1e-3); t.add_argument("--model", default="cinder_lgbm.txt")
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("eval"); e.add_argument("--cache", required=True); e.add_argument("--model", default="cinder_lgbm.txt")
    e.add_argument("--labels", default=None); e.add_argument("--out", default="cinder_submission.csv"); e.set_defaults(func=cmd_eval)
    args = p.parse_args(); args.func(args)

if __name__ == "__main__":
    main()