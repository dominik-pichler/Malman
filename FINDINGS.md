# Cinder — Findings, Lessons & Score Log

The full story of what I learned building a malware detector for the Cinder / EMBER2024
challenge: every score, every bug, every wrong turn, and the corrections. First-person,
written so the reasoning survives — including the mistakes, because they're the instructive part.

---

## Score log (the headline)

### In-distribution validation (zero-leakage temporal holdout, latest 8 train weeks)

| slice | PR-AUC | ROC | TPR@1e-3 |
|---|---:|---:|---:|
| ALL | 0.9979 | 0.9977 | 0.880 |
| Win32 | 0.9978 | 0.9976 | 0.895 |
| Win64 | 0.9985 | 0.9984 | 0.884 |
| Dot_Net | 0.9973 | 0.9972 | 0.777 |

Leakage audit: **0.00%** exact-`sha256` overlap, **0.09%** identical-TLSH overlap between
train- and valid-weeks → the 0.998 is honest, not near-duplicate inflation.

### Challenge grader (the real target — labels held by grader)

| # | model | PR-AUC | ROC | TPR@1%FPR | what changed |
|---|---|---:|---:|---:|---|
| 1 | Windows-only | 0.105 | 0.742 | 0.128 | first passing submission |
| 2 | all-format (naive, overfit) | 0.038 | 0.604 | 0.024 | training on non-PE *hurt* |
| 3 | + extraction-crash fix | 0.089 | 0.577 | 0.115 | ranking no longer **inverted** |
| — | reference recipe | *pending* | | | the actual fix (see below) |

**Known target: other teams exceed 0.4.** So the eval is winnable and our ~0.09 means the
*model* underperforms — not that the problem is capped. (I initially concluded "ceiling
reached"; that was wrong — see Lessons.)

---

## What the challenge actually is

- **Dataset: EMBER2024** (identified from `sample_id: ember2024-...`). Gzipped JSONL,
  **sharded by file type**. Pre-extracted static features, not raw binaries.
- **Multi-format**, despite the "Windows Malware" framing. Train shards:
  win32(48), win64(16), dot_net(8), apk(7), pdf(2), elf(1) = 82 shards, **~5.25M rows**,
  **balanced 50/50 malware/benign within every format**.
- **Timeline**: collected Sep 2023 – Dec 2024. First 52 weeks = train, last 12 weeks = test —
  an explicit "detect malware newer than your training corpus" design.
- **The eval shard is the TEST set, not the adversarial "challenge set."** It's 47,218
  records; the special challenge set is only 6,315 files. So this is a **temporal-drift**
  problem (newer malware), which is very winnable — not an adversarial-evasion wall.
  Eval composition: Win32 27,872 · Win64 9,328 · Dot_Net 4,621 · APK 3,716 · PDF 1,152 ·
  ELF 529 (~89% PE).
- **Features**: the `thrember` extractor produces a **2568-dim** vector per PE record (feature
  version 3). Non-PE files get the byte/string/general subset (PE-structural dims zero); the
  authors state APK/ELF/PDF classifiers *are* trainable from those.
- **Metric**: PR-AUC. The brief also stresses "extremely low FPR" → track TPR@1e-3/1e-4.

---

## The grading environment (this shaped everything)

Verified with `importlib.util.find_spec`: the grader has **numpy, scipy, scikit-learn,
pandas** — but **NO lightgbm, NO pefile, NO thrember** — and refuses `pip install`
(`externally-managed-environment`), Python 3.12.3. Consequences:

- The feature extractor had to be **vendored** (`ember_features.py`).
- The LightGBM model had to be evaluated in **pure numpy** by parsing its text dump —
  verified identical to `lgb.predict` to ~1e-16.
- `pefile`/`signify` are optional (only needed to parse raw bytes, which we never do).
- Submission bundle = `solution.py` + `ember_features.py` + `pefile_warnings.txt` +
  `cinder_lgbm.txt` (self-contained, no installs).

The grader passes **one path** to the eval JSONL and expects **one score per record**
(47,218), matched positionally.

---

## Bugs found & fixed (each mattered)

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | libomp missing (macOS) | lightgbm import `OSError libomp.dylib` | `brew install libomp` |
| 2 | `pe: pefile.PE \| None` annotation evaluated at import | "Error while loading function" with pefile absent | `from __future__ import annotations` |
| 3 | no lightgbm in grader | can't load model | pure-numpy text-model evaluator (==lgb to 1e-16) |
| 4 | **NaN mis-routing** | zero-histograms → 0/0=NaN routed by `≤` (wrong by ≤0.999/record) | parse `decision_type` (default-left + missing-type) |
| 5 | gzip samples | non-JSON bytes → wrong branch | gunzip transparently |
| 6 | one-path-many-records | "Expected 47218 scores, got 1" | iterate records within each file |
| 7 | **crash-zeroing** | unknown pefile-warning/regex/dir-name → `KeyError` → **whole vector zeroed** | tolerant `.get()` lookups |
| 8 | extractor drift risk | train (installed thrember) vs inference (vendored) could differ | `cinder_ember.py` imports the vendored extractor (verified 0 diff) |

**#4 and #7 were the big ones.** #7 in particular: malformed/evasive PEs (exactly the
eval's malware) triggered unknown warnings → the extractor crashed → training **dropped**
those records and inference **zeroed** them → malware scored *benign*, inverting the eval
ROC below 0.5. Fixing it un-inverted the ranking (0.038 → 0.089).

---

## Lessons & wrong turns (the honest part)

1. **"Windows-only" was wrong** — the data and eval are multi-format. Cost a bad first
   submission (0.105) where non-PE flooded the ranking top.
2. **"Training on non-PE hurts" was a red herring** — the 0.038 all-format result came from
   an overfit model *plus* the crash-zeroing bug still present. Non-PE is learnable (the
   authors say so); we later stopped flooring it.
3. **"0% TLSH overlap ⇒ unwinnable ceiling" was my worst call.** Exact-TLSH match is far too
   strict a bar for "catchable"; malware shares family/behavioral patterns without an
   identical fuzzy hash. Other teams > 0.4 prove the signal is there. I over-read one
   diagnostic and nearly wrote off a solvable problem.
4. **The actual root cause: a wildly overfit, mis-configured model.** We ran
   **3000 iterations × 200 leaves, no L2 regularization, trained on weeks 0–43 only, PE-only**.
   That memorizes in-distribution malware (0.998) and **collapses to near-0 on anything
   unfamiliar** — the near-constant eval scores (`p90 = 0.000`) were the tell.
5. **Meta-lesson — go to the reference implementation first.** EMBER2024 ships the exact
   tuned recipe (`examples/lgbm_config.json`, `train_lgbm.py`, 14 benchmark models). Once the
   pipeline worked, I should have replicated that immediately instead of improvising
   hyperparameters and chasing symptoms. The `>0.4` teams didn't find secret signal — they
   used the standard recipe and I didn't.

---

## The reference recipe (current best fix, pending a submit)

From the authors' `lgbm_config.json` + `train_model()` — how it differs from what we ran:

| | reference | ours (wrong) |
|---|---|---|
| num_iterations | **500** | 3000 |
| num_leaves | **64** | 200 |
| learning_rate | 0.1 | 0.05 |
| lambda_l2 | **1.0** | 0 |
| training data | **all 52 weeks**, random 90/10 split | weeks 0–43 only |
| formats | **all**, one model | PE only, non-PE floored |

So the fix (`train_reference.py`): a small, **regularized** model on **all weeks** and
**all formats**, no non-PE floor. Small + L2 = generalizes to newer malware instead of
memorizing. Kept the reference's categorical-feature declaration *off* so the pure-numpy
inference evaluator stays exact.

**Pre-submit signal (no labels needed):** the eval score *distribution*. Our overfit model
gave PE `p90 = 0.000` (near-constant benign). The reference-style model should give **spread**
— a real range with some PE and non-PE near 1. If it spreads, submit; if still floored,
something else is wrong.

---

## Diagnostic tooling built (reusable)

| tool | answers |
|---|---|
| `inspect_schema.py` | what's in a data folder (keys, label balance, time range) |
| `check_submission.py` | count OK? numpy↔lightgbm parity? proxy score? per-format distribution? |
| `analyze.py` | operating points, feature importance by group, error inspection, **leakage audit**, drift |
| `diagnose_shift.py` | vendored-vs-installed extractor parity, train↔eval feature shift, field richness |
| `winnable.py` | does the model catch eval malware resembling training malware? (TLSH twin test) |
| `train_and_probe.py` | train + score eval in one process (no file staleness); `--drop-groups` |
| `train_reference.py` | the benchmark recipe (all weeks/formats, regularized) |

Two process rules learned the hard way: **file staleness** bit us repeatedly (always
`ls -la` / `grep -c '^Tree='` to confirm which model is scored), and the grader is the only
score oracle **and carries penalties** — so every fix is verified offline (`check_submission`)
before spending a submission.

---

## Open items / next

- [ ] Run `train_reference.py`, confirm the eval distribution **spreads**, submit. Expected
      to move well off 0.09 (plausibly 0.2–0.5).
- [ ] If it spreads but underwhelms: add categorical features back (extend the numpy
      evaluator to handle categorical splits so parity holds), grid-search around the config
      with `TimeSeriesSplit` like the authors.
- [ ] Ground truth: `thrember.download_models()` gives the 14 benchmark models — score one on
      the eval directly to see the target a standard model hits.
- [ ] Confirm the actual competitive target (a leaderboard/reference) so effort is calibrated.