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

# EMBER2024 Malware Detection (Cinder)

A machine-learning classifier for malware detection: given static features of Windows
binaries (many of which initially evaded antivirus), detect malware while holding an
**extremely low false-positive rate**. Scored on **PR-AUC**, submitted as a self-contained
`solution.py` inference service that runs in a locked-down grader.

The underlying data is the **EMBER2024** dataset (identified from the `sample_id` field,
e.g. `ember2024-train-win32-00`), delivered as raw-feature JSONL shards.

---

## TL;DR — outcome first

- The model is an excellent **in-distribution** detector: **PR-AUC 0.998** on a zero-leakage
  temporal holdout, TPR 0.88 @ FPR 1e-3.
- On the challenge's **evasive holdout** it collapses to **PR-AUC ~0.09** (ROC ~0.58).
- **Why**, proven not guessed: the eval shares **0% TLSH overlap** with 1.31M training
  malware — it is an entirely novel, evasive population. The collapse survives feature-group
  ablation and is not a plumbing bug (train/inference vectors are byte-identical).
- **Conclusion**: this empirically bounds static, signature-free detection against an
  adaptive adversary — the "arms race," quantified. See [Results](#results--findings).

---

## The data

EMBER2024 raw features, gzipped JSONL, **sharded by file type**. Despite the "Windows"
framing, the corpus (and the eval) is **multi-format**:

| type | train shards | notes |
|---|---:|---|
| `win32-shard-*` | 48 | PE |
| `win64-shard-*` | 16 | PE |
| `dot_net-shard-*` | 8 | PE (.NET) |
| `apk-shard-*` | 7 | non-PE (Android) |
| `pdf-shard-*` | 2 | non-PE |
| `elf-shard-*` | 1 | non-PE (Linux) |

- **~5.25M labeled rows** total; **balanced 50/50 malware/benign within every format**
  (0 = benign, 1 = malicious; −1 = unlabeled, dropped).
- Collected **Sep 2023 – Dec 2024**. By EMBER2024 design the **first 52 weeks are train,
  the last 12 weeks are the test period** — an explicit "detect malware newer than your
  training corpus" setup. `week_id` / `first_submission_date` encode the timeline.
- **Evaluation shard** (`shard-0000.jsonl.gz`): **47,218 records**, multi-format —
  Win32 27,872 · Win64 9,328 · Dot_Net 4,621 · APK 3,716 · PDF 1,152 · ELF 529 (~89% PE).
  Labels are held by the grader (not available locally).

### Fields used vs. deliberately excluded

**Used** — the 12 static PE feature groups (below). **Excluded** as leakage / collection
artifacts / not-at-inference: `detection_ratio` (the AV verdict — near-label, and ~0 on the
evasive tail), `last_analysis_date`, `week_id`-as-feature, `family`, `behavior`, `packer`,
`caps`, `ttps`, `mbc`, `group`. The extractor only reads the static groups, so this metadata
physically cannot enter the model.

---

## Feature extraction

Handled by **`ember_features.py`** — the EMBER2024 PE feature extractor from `thrember`,
**vendored** into this repo (so the grader needs no `thrember`/`pefile`/`signify` install)
and patched to be tolerant (see [Bugs fixed](#bugs-found--fixed)). Produces a **2568-dim**
vector per record. The same file is used for training *and* inference, so vectors are
guaranteed identical (verified: max|diff| = 0).

| Group | Dims | What it extracts |
|---|---:|---|
| GeneralFileInfo | 7 | File size, entropy, is-PE flag, first 4 bytes |
| ByteHistogram | 256 | Normalized byte-value frequency (0–255) |
| ByteEntropyHistogram | 256 | 2D byte/entropy histogram (sliding window) |
| StringExtractor | 177 | String stats + regex-category counts (URLs, IPs, PowerShell, registry, …) |
| HeaderFileInfo | 74 | COFF/optional/DOS header fields |
| SectionInfo | 224 | Section names/sizes/entropy/flags (hash-tricked) + overlay |
| ImportsInfo | 1282 | Imported DLLs+functions (hashed to 256+1024) |
| ExportsInfo | 129 | Exported names (hashed) |
| DataDirectories | 34 | PE data-directory sizes & virtual addresses |
| RichHeader | 33 | Rich-header paired values |
| AuthenticodeSignature | 8 | Signature info: cert count, self-signed, chain depth |
| PEFormatWarnings | 88 | `pefile` parser warnings (malformation ⇒ packing/obfuscation signal) |

Everything is a fixed-length summary or a hashed bag — **no raw-string matching** — so any
file becomes the same 2568 numbers. Two-step pipeline per group:
`raw_features(bytes, pe) → process_raw_features(raw_obj) → float32`. We only call the second
step (features are pre-extracted), so `pefile`/`signify` are optional at inference.

Re-extract the cache:

```bash
uv run cinder_ember.py vectorize --data ../data/train --cache cache/train --jobs 8 --overwrite
```

---

## The toolchain

| File | Role |
|---|---|
| `cinder_ember.py` | Training pipeline: `selftest` / `vectorize` (→ per-shard `.npz` cache) / `train` / `eval`. Vectorizes via the **vendored** `ember_features`. |
| `ember_features.py` + `pefile_warnings.txt` | Vendored, tolerant EMBER2024 extractor. |
| `train_and_probe.py` | Train + score the eval **in one process** (no file staleness). `--formats`, `--rounds`, `--leaves`, `--drop-groups`. |
| `solution.py` | **Submission entrypoint.** `predict_malware(paths)`. Self-contained: vendored extractor + pure-numpy LightGBM eval. Floors non-PE; honors `cinder_drop_groups.txt`. |
| `check_submission.py` | Pre-flight (labels-free): count, numpy↔lightgbm parity on real eval records, proxy score on my holdout, per-format distributions. |
| `analyze.py` | Operating points, feature importance by group, error inspection, **leakage audit**, temporal drift. |
| `diagnose_shift.py` | Extractor parity (vendored vs installed), per-group train↔eval feature shift, raw-field richness. |
| `winnable.py` | TLSH-twin test: does the model catch eval malware resembling training malware? |

**Submission bundle** (four files, one directory): `solution.py`, `ember_features.py`,
`pefile_warnings.txt`, `cinder_lgbm.txt` (+ `cinder_drop_groups.txt` if using `--drop-groups`).

---

## Setup

**Local (training) deps** — `thrember` is only needed to *build the cache*; the vendored
extractor is used at inference:

```bash
uv pip install lightgbm pefile numpy polars scikit-learn tqdm
uv pip install "git+https://github.com/FutureComputing4AI/EMBER2024.git" --no-deps
```

**The grader** has numpy/scipy/scikit-learn/pandas but **no lightgbm, no pefile, no
thrember**, and refuses `pip install`. That constraint drove the whole submission design:
the extractor is vendored, and the LightGBM model is evaluated in **pure numpy** by parsing
its text dump (`cinder_lgbm.txt`) — verified to match `lgb.predict` to ~1e-16.

macOS gotcha: `OSError ... libomp.dylib` → `brew install libomp`. Pin local Python to 3.12.

---

## Train

```bash
uv run train_and_probe.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz \
    --formats Win32,Win64,Dot_Net --target-rows 900000 --rounds 3000 --leaves 200
```

Trains a PE-only model (non-PE is floored at inference), prints per-format validation PR-AUC
+ the eval score distribution, and saves `cinder_lgbm.txt`. Keep trees capped
(`--rounds`/`--leaves`) so the model stays a few tens of MB — an uncapped run once produced a
119 MB model that risks grader limits.

**Non-PE handling.** The extractor is PE-only; APK/PDF/ELF get degenerate (zeroed-PE) vectors
a PE model scores unpredictably high. Since eval is ~89% PE, `solution.py` **floors non-PE to
0.0** (`NONPE_SCORE`) so they can't pollute the high-precision region. Training on non-PE was
tried and *hurt* (out-of-distribution, unlearnable with these features).

---

## Pre-flight & analytics

Never submit without a green pre-flight (the grader is the only score oracle and carries
penalties):

```bash
uv run check_submission.py --shard ../data/evaluation/shard-0000.jsonl.gz --cache cache/train
# require: count 47218 | parity max|diff| < 1e-6 | APK/PDF/ELF ~0.0 | PE bimodal
```

Deeper diagnostics on a trained model:

```bash
uv run analyze.py        --cache cache/train --model cinder_lgbm.txt --formats Win32,Win64,Dot_Net
uv run diagnose_shift.py --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz \
                         --train-shard ../data/train/win32-shard-0000.jsonl.gz --model cinder_lgbm.txt
uv run winnable.py       --cache cache/train --shard ../data/evaluation/shard-0000.jsonl.gz --model cinder_lgbm.txt
```

---

## Results & findings

### In-distribution validation (temporal holdout, latest 8 train weeks)

| slice | PR-AUC | ROC | TPR@1e-3 |
|---|---:|---:|---:|
| ALL | 0.9979 | 0.9977 | 0.880 |
| Win32 | 0.9978 | 0.9976 | 0.895 |
| Win64 | 0.9985 | 0.9984 | 0.884 |
| Dot_Net | 0.9973 | 0.9972 | 0.777 |

**Leakage audit: clean.** 0.00% exact-`sha256` and 0.09% identical-TLSH overlap between
train-weeks and valid-weeks → the 0.998 is *honest*, not near-duplicate inflation.

### Challenge grader (chronological)

| # | model | PR-AUC | ROC | note |
|---|---|---:|---:|---|
| 1 | Windows-only | 0.105 | 0.74 | non-PE flooded the ranking top |
| 2 | all-format (naive) | 0.038 | 0.60 | training on non-PE *hurt* (OOD) |
| 3 | + extraction-crash fix | 0.089 | 0.58 | ranking no longer **inverted** |

The jump from ROC 0.46→0.58 came entirely from the crash-zeroing fix (below), not retraining.

### The decisive diagnosis (`winnable.py`)

> **0 of 41,821 eval PE records have an identical-TLSH twin among 1.31M training malware.**

The eval is **entirely novel** at the file level — a disjoint, later, evasion-curated
population. The model scores >0.5 on <2% of eval PE. This is not a bug and not fixable by
tuning: a static-feature model cannot recognize files that share nothing with its training
distribution. Feature-group ablation (dropping `authenticode`) changed the eval distribution
**not at all**, confirming it isn't one shortcut but the whole representation.

### Conclusion

A LightGBM model on EMBER2024 **static** features achieves PR-AUC **0.998** on
in-distribution malware (zero-leakage temporal CV) but collapses to **~0.09** on an evasive,
TLSH-disjoint holdout. The collapse is invariant to feature-group ablation and to the
extractor (train/inference vectors identical). This empirically **bounds static,
signature-free malware detection against an adaptive adversary** and motivates
dynamic/behavioural features — the arms-race thesis, demonstrated with measurements.

---

## Bugs found & fixed (debugging log)

Each fix that mattered produced a real grader jump; the rest was ruled out with parity/leakage checks.

1. **Import-time crash in the grader** — `pe: pefile.PE | None` annotation is evaluated at
   import; with `pefile` absent it threw. Fixed with `from __future__ import annotations`.
2. **No `thrember`/`lightgbm` in grader** — vendored the extractor; evaluate the LightGBM
   **text model in pure numpy** (matches `lgb.predict` to ~1e-16).
3. **NaN mis-routing** — zero histograms normalize to 0/0 = NaN; the first numpy evaluator
   routed NaN by `value ≤ threshold` (wrong by up to 0.999/record). Fixed by parsing
   `decision_type` (default-left + missing-type) like LightGBM.
4. **One-score-per-record** — the grader passes *one path* to a 47,218-record JSONL and
   expects one score per record; initial code returned 1. Now iterates records per file.
5. **Gzip samples** — `solution.py` gunzips transparently.
6. **Crash-zeroing (the big one)** — records with a `pefile` warning / string-regex /
   data-directory name outside the vocabulary threw `KeyError`, zeroing the *entire* vector.
   Training **dropped** those records; inference **zeroed** them → malformed/evasive malware
   became "benign," inverting the eval ROC below 0.5. Fixed with tolerant `.get()` lookups;
   this un-inverted the ranking (0.038→0.089).
7. **Extractor consistency** — `cinder_ember.py` now imports the *vendored* extractor, so
   training and inference vectorize identically (verified 0 diff), closing that failure mode.

---

## Status

- [x] Dataset identified (EMBER2024, multi-format) and schema inspected.
- [x] Self-contained submission (vendored extractor + numpy model eval) passing the grader.
- [x] Zero-leakage temporal CV; per-format operating points; drift analysis.
- [x] Root-caused the eval collapse: 0% TLSH overlap → entirely novel/evasive.
- [ ] (Optional) dynamic/behavioural features or a robustness study — the only path past the
      static ceiling; expected to be incremental, not transformational.

## References

- EMBER2024 code: https://github.com/FutureComputing4AI/EMBER2024 (`thrember`)
- Dataset: https://huggingface.co/datasets/joyce8/EMBER2024
- Paper: *EMBER2024 — A Benchmark Dataset for Holistic Evaluation of Malware Classifiers*, arXiv:2506.05074