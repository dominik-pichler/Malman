```
        _.---._
     .-' ((O)) '-.
      \ _.\_/._ /
       /..___..\
       ;-.___.-;
      (| e ) e |)     .;.
       \  /_   /      ||||
       _\__-__/_    (\|'-|
     /` / \V/ \ `\   \ )/
    /   \  Y  /   \  /=/
   /  |  \ | / {}  \/ /
  /  /|   `|'   |\   /
  \  \|    |.   | \_/
   \ /\    |.   |
    \_/\   |.   |
    /)_/   |    |
   // ',__.'.__,'
  //   |   |   |
 //    |   |   |
(/     |   |   |
       |   |   |
       | _ | _ |
       |   |   |
       |   |   |
       |   |   |
       |___|___|
       /  J L  \
      (__/   \__)
```

# Cinder — EMBER2024 Malware Detection

A machine-learning classifier for the **Cinder** challenge: given static features of
Windows binaries (many of which initially evaded antivirus), detect malware while
holding an **extremely low false-positive rate**. Scored on **PR-AUC**.

The underlying data is the **EMBER2024** dataset (identified from the `sample_id`
field, e.g. `ember2024-train-win32-00`), delivered as raw-feature JSONL shards.

---

## TL;DR

- Vectorize the raw features with the official **`thrember`** extractor → 2568-dim PE vectors.
- Validate on **time** (`week_id`), never random k-fold — the hidden test is a *later* period.
- Optimize **ranking** for PR-AUC, but track **TPR@low-FPR** (the stated goal) at every slice.
- Feed the model **static PE features only**; keep AV/analysis metadata out (leakage).
- Labels are ~50/50, so the hard part is **drift + the low-FPR tail**, not class imbalance.

---

## The data

EMBER2024 raw features, gzipped JSONL, sharded by architecture:

| | shards | rows/shard | approx total |
|---|---|---|---|
| `win32-shard-*.jsonl.gz` | 29 | ~65,536 | ~1.9M |
| `win64-shard-*.jsonl.gz` | 10 | ~65,536 | ~0.65M |

- **~2.55M labeled records**, label balance ≈ 50/50 (0 = benign, 1 = malicious; −1 = unlabeled, dropped).
- Collected **Sep 2023 – Dec 2024**. By EMBER2024 design, the **first 52 weeks are train,
  the last 12 weeks are test** — an explicit "detect malware newer than your training
  corpus" setup. `week_id` / `first_submission_date` encode the timeline.
- Each record has 33 top-level fields. Only the **static PE feature groups** are used for modeling.

### Fields used vs. deliberately excluded

**Used (via `thrember`, feature version 3):** `histogram`, `byteentropy`, `strings`,
`general`, `header` (DOS/COFF/optional), `section`, `imports`, `exports`,
`datadirectories`, `richheader`, `authenticode`, `pefilewarnings`.

**Excluded — and why:**

- `detection_ratio` — the AV detection count. Near-identical to the label (leakage),
  **and** ~0 on the evasive tail the challenge targets, so it would inflate CV while doing
  nothing for the cases that matter.
- `last_analysis_date`, `week_id` (as a *feature*) — collection artifacts / time shortcuts.
- `family`, `family_confidence`, `behavior`, `packer`, `exploit`, `caps`, `ttps`, `mbc`,
  `group` — analysis-derived tags (mostly empty in these shards) that leak or aren't
  available at inference.

Using the official vectorizer is what enforces this: it only reads the static groups,
so the metadata physically cannot enter the model.

---

## Approach & design decisions

1. **The metric governs everything — and the scored metric ≠ the stated goal.**
   The leaderboard is PR-AUC (the whole ranking curve); the brief asks for an *extremely
   low FPR* (one high-precision operating point). These pull in different directions, so
   the pipeline reports **both** PR-AUC and **TPR@{1e-2, 1e-3, 1e-4}** at every slice.
   Get ranking right first; calibrate and pick the threshold last, as a separate step.

2. **Temporal validation.** I hold out the latest weeks of the training set
   (`VALID_WEEKS`, default 8) to mirror the real train→test gap. Random k-fold would leak
   the future into the past and produce a CV number that collapses on the true test set.

3. **Leakage firewall.** Static PE features only (see table above). Exact-duplicate
   control by `sha256`; near-duplicate / "family" grouping is available via `tlsh`
   (planned slice) since the `family` field is null in these shards.

4. **Not an imbalance problem.** With ~50/50 labels, `scale_pos_weight` ≈ 1 and SMOTE-style
   resampling is irrelevant. Effort goes to the drift tail and the low-FPR operating point.

5. **Slice auditing.** Aggregate PR-AUC can look excellent while the model whiffs on the
   newest weeks or the evasive tail. I slice validation by time and by architecture
   (win32/win64), with a novel-family (TLSH-clustered) slice planned.

---

## Repository files

| File | Purpose |
|---|---|
| `inspect_schema.py` | Stdlib-only schema inspector for the JSONL shards: keys/types, label balance, time range, join key. Run first on any new folder. |
| `cinder_ember.py` | Main pipeline: `vectorize` (thrember → per-shard `.npz` cache), `train` (temporal split + LightGBM + slice metrics), `eval` (score eval shard, join labels, write submission). |
| `cinder_baseline.py` | Format-agnostic imbalanced-detection scaffold (dedup, temporal/novel-family CV, calibration). Kept as a generic reference; `cinder_ember.py` is the EMBER-specific instantiation. |

---

## Setup

`thrember`'s dependency chain (`signify` → `oscrypto`) commonly fails to install on
modern OpenSSL / Colab. I **stub `signify`** in `cinder_ember.py` because I vectorize
*pre-extracted* features — signatures are never parsed from raw bytes — so the native
stack is not needed.

```bash
uv pip install lightgbm pefile numpy polars scikit-learn tqdm
uv pip install "git+https://github.com/FutureComputing4AI/EMBER2024.git" --no-deps
```

---

## Usage

```bash
# 1. Inspect a data folder (always do this first on new data)
uv run python inspect_schema.py --train ../../data/train

# 2. Vectorize raw features into per-shard .npz caches (restartable; ~670 MB/shard)
uv run python cinder_ember.py vectorize --data ../../data/train --cache cache/train
uv run python cinder_ember.py vectorize --data ../../data/eval  --cache cache/eval

# 3. Train (subsample for fast iteration; drop --max-rows for final full-data runs)
uv run python cinder_ember.py train --cache cache/train --max-rows 800000

# 4. Evaluate + write submission (joins external labels by sha256 if needed)
uv run python cinder_ember.py eval --cache cache/eval --labels ../../data/eval/labels.jsonl.gz
```

### Compute notes
- Vectorization only re-hashes the pre-extracted groups (it does **not** re-parse
  binaries), but ~2.55M rows single-process is slow — it parallelizes trivially per shard.
- Full dense matrix ≈ 2.55M × 2568 × 4 B ≈ **26 GB**. Use `--max-rows` to iterate on a
  subsample (fits Colab / smaller RAM); run full only for final candidates (48 GB box).
- LightGBM bins to `uint8` (`max_bin=255`), so the trained representation is far smaller
  than the dense float32 cache.

---

## Status

- [x] Dataset identified (EMBER2024) and schema inspected.
- [x] Vectorization wired to `thrember` (2568-dim confirmed) with the signify workaround.
- [x] Temporal-CV + LightGBM + PR-AUC / TPR@low-FPR reporting + eval scoring, smoke-tested
      end-to-end on synthetic vectors.
- [ ] Real vectorization + training run on the full data (pending, runs locally).
- [ ] Confirm eval schema, `sha256` join, and eval week range via the inspector.
- [ ] Confirm the challenge's exact **submission format** (currently `sha256,score`).

## Planned next steps

- **Calibration + threshold selection** (isotonic) tuned for the low-FPR operating point.
- **Novel-family slice** via TLSH clustering — measure detection on genuinely unseen
  malware, not the easy bulk.
- **Benchmark comparison**: `thrember.download_models()` provides the 14 reference
  EMBER2024 LightGBM classifiers — score the PE detector on the eval cache to get a target
  PR-AUC and see how much headroom remains.
- **Parallelized vectorizer** (multiprocessing across shards).
- Light LightGBM tuning; optional per-architecture models or an `is_win64` feature;
  late ensembling for the final increment.

---

## References

- EMBER2024 code: https://github.com/FutureComputing4AI/EMBER2024 (`thrember`)
- Dataset (HuggingFace): https://huggingface.co/datasets/joyce8/EMBER2024
- Paper: *EMBER2024 — A Benchmark Dataset for Holistic Evaluation of Malware Classifiers*, arXiv:2506.05074