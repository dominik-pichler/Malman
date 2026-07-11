"""
solution.py -- Cinder inference service.

Implements:
    predict_malware(sample_paths: List[Path]) -> List[float]
returning P(malware) in [0, 1] for each sample, in the same order.

Each path is auto-detected:
  * starts with '{'/'['  -> a pre-extracted EMBER2024 raw-feature record (JSON);
                            vectorized with process_raw_features (exact match to training).
  * otherwise            -> a raw binary; vectorized from bytes with feature_vector
                            (pefile parses PE; non-PE -> PE groups zero out, like training).

For submitting the solution, one must bundle these three files together (same directory):
    solution.py, cinder_lgbm.txt, cinder_lgbm.txt.op.pkl

signify note: pre-extracted features never need signify. Raw *signed-PE* extraction
does (8 of 2568 dims). If signify isn't importable, it's stubbed and those dims zero
out -- a tiny mismatch, only relevant if sample_paths are raw PE binaries. Install a
working signify to avoid even that.
"""
from pathlib import Path
from typing import List
import json, pickle, sys, types
import numpy as np

_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "cinder_lgbm.txt"
_OP_PATH = _HERE / "cinder_lgbm.txt.op.pkl"

# --- make signify optional so the module imports anywhere ---
try:
    from signify.authenticode import SignedPEFile as _Probe  # exactly what thrember imports
    _SIGNIFY_OK = True
except Exception:
    _SIGNIFY_OK = False
    _a = types.ModuleType("signify.authenticode"); _a.SignedPEFile = object
    _e = types.ModuleType("signify.exceptions")
    _e.SignerInfoParseError = _e.ParseError = type("_E", (Exception,), {})
    sys.modules["signify"] = types.ModuleType("signify")
    sys.modules["signify.authenticode"] = _a
    sys.modules["signify.exceptions"] = _e

from thrember.features import PEFeatureExtractor
if not _SIGNIFY_OK:
    # never call signify during raw-byte extraction; zero the authenticode group instead
    from thrember.features import AuthenticodeSignature
    AuthenticodeSignature.raw_features = lambda self, bytez, pe: {}

import lightgbm as lgb

_EX = PEFeatureExtractor()
_MODEL = lgb.Booster(model_file=str(_MODEL_PATH))
_OP = pickle.load(open(_OP_PATH, "rb")) if _OP_PATH.exists() else {}
_ISO = _OP.get("iso")
_DIM = _EX.dim


def _vectorize(path) -> np.ndarray:
    data = Path(path).read_bytes()
    head = data[:64].lstrip()
    if head[:1] in (b"{", b"["):                       # pre-extracted feature record
        return np.asarray(_EX.process_raw_features(json.loads(data)), dtype=np.float32)
    return np.asarray(_EX.feature_vector(data), dtype=np.float32)  # raw binary


def predict_malware(sample_paths: List[Path]) -> List[float]:
    X = np.zeros((len(sample_paths), _DIM), dtype=np.float32)
    for i, p in enumerate(sample_paths):
        try:
            X[i] = _vectorize(p)
        except Exception:
            pass  # unparseable -> zero vector; model still scores it, batch never fails
    raw = _MODEL.predict(X)
    prob = _ISO.predict(raw) if _ISO is not None else raw
    return [float(x) for x in np.asarray(prob).ravel()]


if __name__ == "__main__":  # quick manual check: python solution.py <sample> [<sample> ...]
    paths = [Path(a) for a in sys.argv[1:]]
    for p, s in zip(paths, predict_malware(paths)):
        print(f"{s:.4f}  {p}")