"""
Cinder schema inspector — run locally, paste the output back.

Usage:
    python inspect_schema.py --train ../data/train --eval ../data/eval

Prints, using only the stdlib (safe on Colab):
  * top-level keys of a train record + a type/shape descriptor for each value
  * candidate label / timestamp / id fields it auto-detects
  * label balance + timestamp range over a sample
  * win32 vs win64 shard inventory
  * eval: schema of shard-0000 + labels.jsonl.gz, and the key they join on
"""
import argparse, glob, gzip, json, os
from collections import Counter

LABEL_HINTS = ("label", "y", "malicious", "is_malware", "malware", "target", "class")
TIME_HINTS  = ("appeared", "timestamp", "first_seen", "firstseen", "date", "build",
               "compile", "seen", "time")
ID_HINTS    = ("sha256", "sha1", "md5", "hash", "id", "uuid", "name")


def describe(v, depth=0):
    """Compact type/shape descriptor, depth-limited so EMBER-style nesting is legible."""
    if isinstance(v, bool):   return "bool"
    if isinstance(v, int):    return "int"
    if isinstance(v, float):  return "float"
    if isinstance(v, str):    return f"str[{len(v)}] e.g. {v[:24]!r}"
    if isinstance(v, list):
        inner = describe(v[0], depth + 1) if v else "?"
        return f"list[{len(v)}] of {inner}"
    if isinstance(v, dict):
        if depth >= 2:
            return f"dict{{{len(v)} keys}}"
        return "dict{ " + ", ".join(f"{k}: {describe(val, depth+1)}"
                                    for k, val in list(v.items())[:12]) + " }"
    return type(v).__name__


def read_records(path, limit):
    out = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def guess(keys, hints):
    kl = {k.lower(): k for k in keys}
    for h in hints:                       # exact first
        if h in kl:
            return kl[h]
    for h in hints:                       # substring fallback
        for low, orig in kl.items():
            if h in low:
                return orig
    return None


def inspect_shard(path, sample=20000):
    print(f"\n=== {path} ===")
    recs = read_records(path, 1)
    if not recs:
        print("  (empty)")
        return
    r = recs[0]
    keys = list(r.keys())
    print(f"top-level keys ({len(keys)}): {keys}")
    print("\nper-field descriptor:")
    for k in keys:
        print(f"  {k:16} : {describe(r[k])}")

    lab = guess(keys, LABEL_HINTS)
    tim = guess(keys, TIME_HINTS)
    idf = guess(keys, ID_HINTS)
    print(f"\ncandidates -> label={lab!r}  time={tim!r}  id={idf!r}")

    # label balance + time range over a larger sample
    labs, times = Counter(), []
    big = read_records(path, sample)
    for x in big:
        if lab in x:  labs[x[lab]] += 1
        if tim in x:  times.append(x[tim])
    if labs:
        print(f"label balance (first {len(big)}): {dict(labs)}")
    if times:
        print(f"{tim} range: {min(times)} .. {max(times)}")
    # how many fields look numeric (already-vectorized signal)
    numeric = [k for k in keys if isinstance(r[k], (int, float, bool))]
    print(f"numeric top-level fields: {len(numeric)}"
          f"{' -> looks ALREADY VECTORIZED' if len(numeric) > 100 else ' -> looks RAW (needs extraction)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train")
    ap.add_argument("--eval",  default="../data/eval")
    args = ap.parse_args()

    # ---- train inventory ----
    tr32 = sorted(glob.glob(os.path.join(args.train, "win32-shard-*.jsonl.gz")))
    tr64 = sorted(glob.glob(os.path.join(args.train, "win64-shard-*.jsonl.gz")))
    print(f"train shards: win32={len(tr32)}  win64={len(tr64)}")
    if tr32: inspect_shard(tr32[0])
    if tr64: inspect_shard(tr64[0])

    # rough row count from one shard (avoid reading all 2.6M)
    if tr32:
        with gzip.open(tr32[0], "rt", errors="replace") as f:
            n = sum(1 for _ in f)
        print(f"\n~{n} records in {os.path.basename(tr32[0])} "
              f"-> ~{n*(len(tr32)+len(tr64)):,} total (rough)")

    # ---- eval ----
    ev_feat = glob.glob(os.path.join(args.eval, "shard-*.jsonl.gz"))
    ev_lab  = glob.glob(os.path.join(args.eval, "labels.jsonl.gz"))
    if ev_feat:
        inspect_shard(ev_feat[0])
    if ev_lab:
        print(f"\n=== {ev_lab[0]} (labels) ===")
        lr = read_records(ev_lab[0], 1)[0]
        print(f"label-file keys: {list(lr.keys())}")
        for k, v in lr.items():
            print(f"  {k:16} : {describe(v)}")
        # find the join key shared with the feature shard
        if ev_feat:
            fk = set(read_records(ev_feat[0], 1)[0].keys())
            shared = fk & set(lr.keys())
            print(f"shared keys with feature shard (join candidates): {sorted(shared)}")


if __name__ == "__main__":
    main()