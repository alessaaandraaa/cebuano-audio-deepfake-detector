r"""
corpus_balance.py

Full accounting of the corpus, on disk and in the manifests, so the
arithmetic in Table 3 can be checked rather than asserted.

Answers, with no inference:

  * how many clips exist on disk per source
  * how many survive into the built splits, and how many were excluded
  * whether bona fide and spoof are balanced OVERALL (they should be)
  * how the spoof half divides between the two synthesis systems
    (they will NOT be equal, and that is expected)

Usage:

    python training\corpus_balance.py
    python training\corpus_balance.py --splits-dir manifests\splits_norm
    python training\corpus_balance.py --audio-root data\processed
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)


def count_disk(root: Path) -> dict:
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir():
            out[d.name] = sum(1 for _ in d.rglob("*.wav"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus balance accounting.")
    ap.add_argument("--splits-dir", default="manifests/splits_vc")
    ap.add_argument("--audio-root", default="data/normalised")
    args = ap.parse_args()

    root = Path(args.audio_root)
    splits = Path(args.splits_dir)

    print()
    print("=" * 66)
    print(f"ON DISK  ({root})")
    print("=" * 66)
    disk = count_disk(root)
    bona_disk = disk.pop("bonafide", 0)
    print(f"  {'bonafide':22s} {bona_disk:7d}")
    for k, v in sorted(disk.items()):
        print(f"  {k:22s} {v:7d}")
    spoof_disk = sum(disk.values())
    print(f"  {'-'*22} {'-'*7}")
    print(f"  {'spoof total':22s} {spoof_disk:7d}")
    print("\n  NOTE: this counts every source present under the audio root,")
    print("  including any that the splits below do not use. The meaningful")
    print("  comparison is made per-source further down, not on this total.")

    print()
    print("=" * 66)
    print(f"IN THE BUILT SPLITS  ({splits})")
    print("=" * 66)
    frames = {}
    for s in ("train", "val", "test"):
        f = splits / f"{s}.parquet"
        if f.exists():
            frames[s] = pd.read_parquet(f)
    if not frames:
        print(f"  no parquets in {splits}")
        return 1

    allrows = pd.concat(frames.values(), ignore_index=True)
    by = Counter(allrows["spoof_method"])
    bona_m = by.pop("none", 0)
    print(f"  {'bonafide':22s} {bona_m:7d}")
    for k, v in sorted(by.items()):
        print(f"  {k:22s} {v:7d}")
    spoof_m = sum(by.values())
    print(f"  {'-'*22} {'-'*7}")
    print(f"  {'spoof total':22s} {spoof_m:7d}")
    print()
    ok = spoof_m == bona_m
    print(f"  BALANCED OVERALL: {'YES' if ok else 'NO'}   "
          f"(bonafide {bona_m} vs spoof {spoof_m}, "
          f"difference {spoof_m - bona_m:+d})")

    if len(by) == 2:
        a, b = sorted(by.items())
        print(f"\n  The two systems are NOT equal to each other, and should "
              f"not be:")
        print(f"    {a[0]:20s} {a[1]:7d}")
        print(f"    {b[0]:20s} {b[1]:7d}")
        print(f"    difference          {abs(a[1]-b[1]):7d}")
        print(f"  Each system covers a DIFFERENT half of the 140 speakers,")
        print(f"  and speakers hold different numbers of clips (median ~441,")
        print(f"  lowest 94). Equal speaker counts, unequal clip counts.")

    print()
    print("=" * 66)
    print("EXCLUDED BY THE PAIRING GUARD")
    print("=" * 66)
    rep = splits / "unpaired_excluded.csv"
    if not rep.exists():
        print(f"  no report at {rep}")
    else:
        rows = list(csv.DictReader(open(rep, encoding="utf-8-sig")))
        print(f"  {len(rows)} clip(s) had no bona fide counterpart\n")
        for (m, spk), n in sorted(Counter(
                (r["spoof_method"], r["speaker_id"]) for r in rows).items(),
                key=lambda kv: -kv[1]):
            print(f"    {m:14s} speaker {spk}   {n:5d}")
        print()
        print("  per-source reconciliation (only sources the splits use):")
        tot_used = 0
        for m in sorted(by):
            on_disk = disk.get(m, 0)
            excl = sum(1 for r in rows if r["spoof_method"] == m)
            tot_used += on_disk - excl
            flag = "ok" if on_disk - excl == by[m] else "MISMATCH"
            print(f"    {m:14s} disk {on_disk:6d} - excluded {excl:4d} "
                  f"= {on_disk - excl:6d}   manifest {by[m]:6d}   {flag}")
        unused = sorted(set(disk) - set(by))
        if unused:
            print(f"\n  present on disk but NOT in these splits: "
                  f"{', '.join(f'{u} ({disk[u]})' for u in unused)}")
        print(f"\n    {'total spoof':14s} {tot_used:6d}   "
              f"bona fide {bona_m:6d}   "
              f"{'BALANCED' if tot_used == bona_m else 'UNBALANCED'}")

    print()
    print("=" * 66)
    print("SPEAKERS")
    print("=" * 66)
    print(f"  bona fide speakers in splits : "
          f"{allrows[allrows.spoof_method == 'none'].speaker_id.nunique()}")
    for m in sorted(by):
        sub = allrows[allrows.spoof_method == m]
        print(f"  {m:28s} : {sub.speaker_id.nunique()} speakers, "
              f"{len(sub)} clips")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
