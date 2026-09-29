r"""
make_vc_pairs.py

Assigns each source speaker a TARGET VOICE for voice-conversion spoof
generation, and proves the assignment is safe before a single clip is
generated.

WHY VOICE CONVERSION

The corpus previously paired ElevenLabs voice cloning with Meta MMS
text-to-speech. MMS produces a SINGLE synthetic voice per language, and
the per-generator analysis specified in section 4.10 found that half of
the spoof class had degenerated into speaker identification:

    per-source EER      meta-mms 0.000237   elevenlabs 0.012254   (52x)
    trained on MMS only, matched condition          EER 0.0000
    trained on MMS only, unseen ElevenLabs          EER 0.6040

0.6040 is BELOW CHANCE. The detector had learned "that one voice is
spoof, everything else is bona fide", so clones wearing real speakers'
voices were confidently filed as genuine.

Voice conversion replaces it because:

  * It is a distinct attack family, not a relabelling. ASVspoof 2019 LA
    lists A05, A06, A17, A18 and A19 as VC, separate from its TTS
    attacks. The training set is four TTS plus two VC; this corpus
    currently has two TTS and no VC.

  * It is TEXT-FREE. Source speech in, converted speech out, no
    pronunciation rules or phoneme set. Cebuano cannot break it -- the
    failure mode that has cost this project the most time.

  * Both confounds die by construction. See below.

THE PAIRING, AND WHY IT IS SHAPED THIS WAY

For each bona fide clip from source speaker A, the audio is converted
to the voice of target speaker B, giving A's words in B's voice.

    the WORDS are A's   -> they appear in bona fide, as A's own clip
    the VOICE is B's    -> it appears in bona fide, as B's recordings

Neither content nor speaker identity carries information about the
label. That is a stricter control than the ElevenLabs half, where only
the content is matched.

Three constraints, all enforced and verified here:

  1. B must be in the SAME SPLIT as A.

     Otherwise a training speaker's voice ends up inside the test set
     (or vice versa) and the speaker-disjoint guarantee in section
     4.3.2 is quietly broken. Nothing downstream would catch this --
     the clip carries A's speaker_id, so the split logic sees nothing
     wrong. It has to be prevented here or not at all.

  2. B must be drawn from the VC half.

     Keeps the two synthesis halves independent, so a
     leave-one-system-out run that holds out VC does not implicitly
     involve ElevenLabs speakers.

  3. B != A. A self-pairing is a no-op conversion.

The assignment is a DERANGEMENT within each split -- a permutation with
no fixed points. Every speaker is a target exactly once, so each voice
appears in the spoof class in proportion to its partner's clip count,
rather than a few voices dominating.

Reference clips are NOT withheld from the bona fide set, unlike the
ElevenLabs half. There the reference WAS the cloned material; here it
only supplies timbre, and the converted output shares no content with
it.

Usage:

    python preprocessing\make_vc_pairs.py            # dry run
    python preprocessing\make_vc_pairs.py --apply
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow",
          file=sys.stderr)
    sys.exit(1)


def load_split_map(splits_dir: Path) -> dict:
    """{speaker_id: split} from the built manifests."""
    out = {}
    for s in ("train", "val", "test"):
        p = splits_dir / f"{s}.parquet"
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)
        for spk in pd.read_parquet(p)["speaker_id"].unique():
            out[str(spk)] = s
    return out


def load_speakers(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            (r.get("speaker_id") or list(r.values())[0]).strip()
            for r in csv.DictReader(f)
        ]


def derange(items: list, rng: random.Random, tries: int = 200) -> dict:
    """
    A permutation with no fixed point.

    Rejection sampling: for n >= 4 roughly 1/e of shuffles are already
    derangements, so this terminates quickly. The cyclic fallback is
    there so the function cannot hang or return an invalid mapping.
    """
    n = len(items)

    if n < 2:
        return {}

    for _ in range(tries):
        shuffled = items[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled)):
            return dict(zip(items, shuffled))

    # Guaranteed derangement: rotate by one.
    return {a: items[(i + 1) % n] for i, a in enumerate(items)}


def check_pairs(pairs: dict, split_of: dict, vc_set: set, root: Path):
    """
    The properties the methodology depends on. Returns (checks, problems).

    Run at build time AND after every manifest rebuild. A pairing that
    crosses a split boundary puts one speaker's voice on both sides of
    the train/test line, and nothing downstream can see it, because the
    converted clip carries the SOURCE speaker's id.
    """
    problems = []

    self_paired = [a for a, b in pairs.items() if a == b]
    if self_paired:
        problems.append(f"{len(self_paired)} self-pairing(s): "
                        f"{self_paired[:5]}")

    unknown = sorted({s for s in pairs if s not in split_of} |
                     {b for b in pairs.values() if b not in split_of})
    if unknown:
        problems.append(f"{len(unknown)} speaker(s) absent from the current "
                        f"splits: {unknown[:5]}")

    crossed = [
        (a, b) for a, b in pairs.items()
        if a in split_of and b in split_of and split_of[a] != split_of[b]
    ]
    if crossed:
        problems.append(f"{len(crossed)} CROSS-SPLIT pairing(s): {crossed[:5]}")

    outside = [(a, b) for a, b in pairs.items() if b not in vc_set]
    if outside:
        problems.append(f"{len(outside)} target(s) outside the VC half")

    no_bona = [b for b in set(pairs.values()) if not (root / b).is_dir()]
    if no_bona:
        problems.append(f"{len(no_bona)} target(s) have no bona fide audio: "
                        f"{no_bona[:5]}")

    repeats = {b: n for b, n in Counter(pairs.values()).items() if n > 1}
    if repeats:
        problems.append(f"not a derangement -- {len(repeats)} target(s) used "
                        "more than once")

    return [
        ("no speaker converted to itself", not self_paired),
        ("every speaker present in the splits", not unknown),
        ("every pair within one split", not crossed),
        ("every target inside the VC half", not outside),
        ("every target has bona fide audio", not no_bona),
        ("derangement (each voice targeted once)", not repeats),
    ], problems


def verify_existing(path: Path, split_of: dict, vc_set: set, root: Path) -> int:
    """Re-check a written mapping against the CURRENT splits."""
    if not path.exists():
        print(f"ERROR: no mapping at {path} -- nothing to verify.",
              file=sys.stderr)
        return 1

    pairs, recorded = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pairs[r["source_speaker"]] = r["target_speaker"]
            recorded[r["source_speaker"]] = r.get("split")

    print(f"  mapping : {path}   ({len(pairs)} pairs)")
    print()

    checks, problems = check_pairs(pairs, split_of, vc_set, root)

    # Did the split assignment itself move since this was written?
    drifted = [(a, recorded[a], split_of[a]) for a in pairs
               if a in split_of and recorded.get(a)
               and recorded[a] != split_of[a]]
    checks.append(("recorded split still matches the manifests", not drifted))
    if drifted:
        problems.append(
            f"{len(drifted)} speaker(s) changed split since the mapping was "
            f"written: {drifted[:3]} -- the audio and the splits now "
            "disagree; regenerate one or the other"
        )

    unpaired = sorted(vc_set - set(pairs))
    checks.append(("every VC-half speaker still paired", not unpaired))
    if unpaired:
        problems.append(f"{len(unpaired)} VC speaker(s) unpaired: "
                        f"{unpaired[:5]}")

    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    print()

    if problems:
        for pr in problems:
            print(f"  FAIL: {pr}", file=sys.stderr)
        print("RESULT: mapping is NOT safe to generate or train from.",
              file=sys.stderr)
        return 1

    print("  Mapping is consistent with the current splits.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=("Assign voice-conversion target speakers, "
                     "within-split and self-excluding.")
    )
    ap.add_argument("--splits-dir", default="manifests/splits_norm")
    ap.add_argument("--vc-speakers", default="manifests/mms_selected_speakers.csv",
                    help="the half being converted (currently the MMS half)")
    ap.add_argument("--bonafide-root", default="data/processed/bonafide")
    ap.add_argument("--out", default="manifests/vc_speaker_pairs.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help=("re-check an EXISTING mapping against the current "
                          "splits instead of building one. Run after every "
                          "manifest rebuild."))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    split_of = load_split_map(Path(args.splits_dir))
    vc = load_speakers(Path(args.vc_speakers))
    root = Path(args.bonafide_root)

    print("=" * 74)
    print("VOICE-CONVERSION SPEAKER PAIRING" +
          ("   (VERIFY)" if args.verify
           else "" if args.apply else "   (DRY RUN)"))
    print("=" * 74)
    print()

    if args.verify:
        return verify_existing(Path(args.out), split_of, set(vc), root)

    missing = [s for s in vc if s not in split_of]
    if missing:
        print(f"ERROR: {len(missing)} speaker(s) absent from the splits: "
              f"{missing[:5]}", file=sys.stderr)
        return 1

    by_split = defaultdict(list)
    for s in vc:
        by_split[split_of[s]].append(s)

    clips = {}
    for s in vc:
        d = root / s
        clips[s] = len([f for f in os.listdir(d) if f.endswith(".wav")]) \
            if d.is_dir() else 0

    rng = random.Random(args.seed)
    pairs = {}

    for split in ("train", "val", "test"):
        group = sorted(by_split.get(split, []))
        if len(group) < 2:
            print(f"ERROR: split {split!r} has {len(group)} VC speaker(s); "
                  "cannot pair within it.", file=sys.stderr)
            return 1
        pairs.update(derange(group, rng))
        print(f"  {split:5} {len(group):3} speakers  "
              f"{sum(clips[s] for s in group):6} clips")

    print()

    # ---------------------------------------------------------------
    # VERIFY -- these are the properties the methodology depends on
    # ---------------------------------------------------------------
    checks, problems = check_pairs(pairs, split_of, set(vc), root)

    unpaired = [s for s in vc if s not in pairs]
    checks.append(("every source speaker paired", not unpaired))
    if unpaired:
        problems.append(f"{len(unpaired)} source(s) unpaired")

    print("  VERIFICATION")
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    print()
    if problems:
        for p in problems:
            print(f"  FAIL: {p}", file=sys.stderr)
        print("RESULT: FAILED -- do not generate from this.", file=sys.stderr)
        return 1

    total = sum(clips[s] for s in vc)
    print(f"  {len(pairs)} pairings, {total} clips to convert")
    print()
    print("  sample (source -> target voice, split, clips):")
    for a in sorted(pairs)[:6]:
        print(f"    {a} -> {pairs[a]}   {split_of[a]:5}  {clips[a]:4}")
    print("    ...")
    print()

    if not args.apply:
        print("Dry run. Re-run with --apply to write the mapping.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_speaker", "target_speaker", "split", "n_clips"])
        for a in sorted(pairs):
            w.writerow([a, pairs[a], split_of[a], clips[a]])

    print(f"Wrote {out}  ({len(pairs)} rows)")
    print()
    print("This file is the reproducibility record for the VC half and")
    print("should be released with the corpus. Generation reads it; do")
    print("not regenerate it with a different seed once clips exist, or")
    print("the audio and the mapping will disagree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
