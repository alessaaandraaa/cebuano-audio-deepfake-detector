r"""
build_manifests.py

Builds the DeepFense Parquet manifests for the Cebuano audio deepfake
dataset, using a 70/15/15 SPEAKER-DISJOINT split stratified by gender.

Split policy (per thesis Section 4.3.2):

    - Partition at the SPEAKER level, not the clip level, so that no
      speaker appears in more than one split. This prevents speaker
      identity leakage.
    - Stratify by gender so the corpus ratio (57.9% female /
      42.1% male) is preserved across train, validation, and test.
    - Fixed random seed for reproducibility.

Expected input layout:

    data/processed/bonafide/<speaker_id>/<name>.wav
    data/processed/meta-mms/<speaker_id>/<name>.1.wav
    data/processed/elevenlabs/<speaker_id>/<name>.2.wav

Speaker gender is read from the bonafide manifest produced by
inventory_bonafide.py (columns: speaker_id, speaker_gender).

Output:

    <out>/train.parquet
    <out>/val.parquet
    <out>/test.parquet
    <out>/split_summary.txt

Parquet columns:

    ID            unique clip identifier
    path          absolute path to the audio file
    label         "bonafide" or "spoof"   (DeepFense label_map)
    speaker_id    for auditing the split
    spoof_method  "elevenlabs" / "meta-mms" / "none"
    duration_sec  written only when --durations is passed

    ID, path, and label are what DeepFense requires. The rest are
    carried for subgroup analysis (per-synthesis-method EER) and for
    verifying the split after the fact.

IMPORTANT:

    Each speaker is assigned to exactly ONE synthesis method upstream
    (half to ElevenLabs, half to Meta MMS). Because the split here is
    speaker-level, a speaker's bonafide clips and their spoof clips
    always land in the SAME split. That is intentional -- the
    alternative would leak the speaker across splits.

    A consequence worth checking in the summary: the two spoof
    methods will not be perfectly balanced within each split, since
    speakers are assigned to methods independently of this split.

Requires:

    pip install pandas pyarrow soundfile

Usage:

    # Full dataset
    python training\build_manifests.py \
        --bonafide-root data\processed\bonafide \
        --spoof-root meta-mms=data\processed\meta-mms \
        --spoof-root elevenlabs=data\processed\elevenlabs \
        --speaker-manifest manifests\manifest_bonafide.csv \
        --out manifests\splits

    # Small smoke-test subset: 6 speakers, 40 clips each
    python training\build_manifests.py \
        --bonafide-root data\processed\bonafide \
        --spoof-root meta-mms=data\processed\meta-mms \
        --spoof-root elevenlabs=data\processed\elevenlabs \
        --speaker-manifest manifests\manifest_bonafide.csv \
        --out manifests\splits_smoke \
        --max-speakers 6 \
        --max-clips-per-speaker 40
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print(
        "Missing dependency. Run: pip install pandas pyarrow",
        file=sys.stderr,
    )
    sys.exit(1)


AUDIO_EXTENSIONS = {".wav", ".flac"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test gets the remainder, so the three always sum to every speaker

LABEL_BONAFIDE = "bonafide"
LABEL_SPOOF = "spoof"


# ---------------------------------------------------------------------------
# SPEAKER METADATA
# ---------------------------------------------------------------------------


def load_speaker_gender(manifest_path: Path) -> dict:
    """Returns {speaker_id: gender}, one entry per unique speaker."""
    speaker_gender = {}

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if (
            "speaker_id" not in reader.fieldnames
            or "speaker_gender" not in reader.fieldnames
        ):
            print(
                "ERROR: manifest needs 'speaker_id' and "
                "'speaker_gender' columns.",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            speaker = row["speaker_id"].strip()
            gender = row.get("speaker_gender", "").strip().lower()

            if speaker and gender and speaker not in speaker_gender:
                speaker_gender[speaker] = gender

    return speaker_gender


# ---------------------------------------------------------------------------
# CLIP COLLECTION
# ---------------------------------------------------------------------------


def collect_clips(
    root: Path,
    label: str,
    spoof_method: str,
    max_clips_per_speaker: int | None,
) -> dict:
    """
    Returns {speaker_id: [clip_row, ...]}.

    Speaker id is taken from the containing folder name, matching the
    layout produced by the generation scripts.
    """
    by_speaker = defaultdict(list)

    if not root.exists():
        print(
            f"WARNING: root not found, skipping: {root}",
            file=sys.stderr,
        )
        return by_speaker

    for speaker_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        speaker_id = speaker_dir.name

        clips = sorted(
            p
            for p in speaker_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )

        if max_clips_per_speaker is not None:
            clips = clips[:max_clips_per_speaker]

        for path in clips:
            by_speaker[speaker_id].append(
                {
                    "ID": path.stem,
                    "path": str(path.resolve()),
                    "label": label,
                    "speaker_id": speaker_id,
                    "spoof_method": spoof_method,
                }
            )

    return by_speaker


# ---------------------------------------------------------------------------
# PAIRING GUARD
# ---------------------------------------------------------------------------
#
# This corpus is a PAIRED design: every spoof clip is generated from the
# text of a specific bonafide utterance, so the two classes differ in
# synthesis and not in what is being said. That control is the main
# defence against a detector learning CONTENT rather than artefacts --
# see Dao et al., "Linguistic Bias Mitigation for Spoofing Detection",
# who needed a gradient-reversal architecture to recover a property this
# corpus has by construction.
#
# A spoof clip whose bonafide counterpart is missing breaks it. It puts a
# sentence into one class only, and it cannot take part in the paired
# comparison it exists for. Worse, it does so invisibly: nothing in the
# split summary or the training logs will ever mention it.
#
# Spoof filenames carry a single-digit synthesis suffix before the
# extension -- <bonafide-stem>.1.wav for Meta MMS, .2.wav for ElevenLabs.
# Bonafide stems end in a FOUR-digit utterance number (0200.111020.
# 092714.0007), so stripping exactly one trailing ".<digit>" recovers the
# counterpart and can never truncate a bonafide stem.

SPOOF_SUFFIX_RE = re.compile(r"\.\d$")


def bonafide_stem_for(spoof_stem: str) -> str:
    """Stem of the bonafide clip a spoof clip was generated from."""
    return SPOOF_SUFFIX_RE.sub("", spoof_stem)


def enforce_pairing(clips_by_speaker: dict, max_unpaired_frac: float):
    """
    Drop spoof clips with no bonafide counterpart for the same speaker.

    Mutates clips_by_speaker. Returns (n_spoof_seen, excluded_rows).

    Matching is per speaker, not global, so a stem collision across
    speakers cannot silently pair a clip with the wrong original.

    Exits if the unpaired fraction exceeds max_unpaired_frac. Past that
    point the likeliest explanation is that the filename convention
    changed, and silently discarding most of the spoof class would do
    far more damage than stopping.
    """
    excluded = []
    n_spoof = 0

    for speaker, rows in clips_by_speaker.items():
        bona = {r["ID"] for r in rows if r["label"] == LABEL_BONAFIDE}

        kept = []

        for row in rows:
            if row["label"] != LABEL_SPOOF:
                kept.append(row)
                continue

            n_spoof += 1

            if bonafide_stem_for(row["ID"]) in bona:
                kept.append(row)
            else:
                excluded.append(row)

        clips_by_speaker[speaker] = kept

    if n_spoof and len(excluded) / n_spoof > max_unpaired_frac:
        pct = 100.0 * len(excluded) / n_spoof
        print(
            f"ERROR: {len(excluded)} of {n_spoof} spoof clips ({pct:.1f}%)"
            " have no bonafide counterpart, above the"
            f" {100 * max_unpaired_frac:.1f}% limit.\n"
            "       This is far more likely to be a filename-convention"
            " mismatch than a real orphan problem.\n"
            "       Expected spoof naming: <bonafide-stem>.<digit>.wav\n"
            "       If the exclusion really is this large, raise"
            " --max-unpaired-frac deliberately.",
            file=sys.stderr,
        )
        sys.exit(1)

    return n_spoof, excluded


def report_unpaired(excluded: list, n_spoof: int, report_path):
    """Print the exclusion breakdown; optionally write the full list."""
    if not excluded:
        print("Pairing guard: every spoof clip has a bonafide counterpart.")
        return

    pct = 100.0 * len(excluded) / n_spoof if n_spoof else 0.0

    print(
        f"Pairing guard: excluded {len(excluded)} of {n_spoof} spoof clips"
        f" ({pct:.2f}%) with no bonafide counterpart."
    )

    by_method = Counter(r["spoof_method"] for r in excluded)

    for method, n in by_method.most_common():
        speakers = Counter(
            r["speaker_id"] for r in excluded if r["spoof_method"] == method
        )
        head = ", ".join(f"{s}:{c}" for s, c in speakers.most_common(5))
        more = " ..." if len(speakers) > 5 else ""
        print(f"    {method:12} {n:6}  across {len(speakers)} speaker(s)")
        print(f"                        {head}{more}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(excluded).sort_values(
            ["spoof_method", "speaker_id", "ID"]
        ).to_csv(report_path, index=False)
        print(f"    full list: {report_path}")


# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------


def split_speakers(
    speakers: list,
    speaker_gender: dict,
    seed: int,
) -> dict:
    """
    Splits speakers 70/15/15 WITHIN each gender group, so the gender
    ratio is preserved across all three splits.

    Returns {speaker_id: "train" | "val" | "test"}.
    """
    by_gender = defaultdict(list)

    for speaker in speakers:
        by_gender[speaker_gender.get(speaker, "unknown")].append(speaker)

    rng = random.Random(seed)
    assignment = {}

    for gender, group in sorted(by_gender.items()):
        # sort before shuffling so the result depends only on the seed
        group = sorted(group)
        rng.shuffle(group)

        n = len(group)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)

        # guarantee val and test are non-empty when the group allows it
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))

        for speaker in group[:n_train]:
            assignment[speaker] = "train"

        for speaker in group[n_train : n_train + n_val]:
            assignment[speaker] = "val"

        for speaker in group[n_train + n_val :]:
            assignment[speaker] = "test"

    return assignment


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------


def build_summary(
    frames: dict,
    assignment: dict,
    speaker_gender: dict,
    seed: int,
) -> str:
    lines = []
    out = lines.append

    out("=" * 70)
    out("SPLIT SUMMARY")
    out("=" * 70)
    out(f"Seed                 : {seed}")
    out(f"Policy               : speaker-disjoint, gender-stratified")
    out(f"Ratios               : 70 / 15 / 15")
    out("")

    total_speakers = len(assignment)
    gender_totals = Counter(
        speaker_gender.get(s, "unknown") for s in assignment
    )

    out(f"Total speakers       : {total_speakers}")
    for gender, count in sorted(gender_totals.items()):
        pct = 100 * count / total_speakers if total_speakers else 0
        out(f"  {gender:8s}: {count:4d}  ({pct:.1f}%)")
    out("")

    for split in ("train", "val", "test"):
        df = frames[split]
        speakers = sorted(s for s, v in assignment.items() if v == split)

        out("-" * 70)
        out(f"{split.upper()}")
        out("-" * 70)
        out(f"  speakers   : {len(speakers)}")
        out(f"  clips      : {len(df)}")

        genders = Counter(speaker_gender.get(s, "unknown") for s in speakers)
        gender_str = ", ".join(
            f"{g}={n} ({100 * n / len(speakers):.1f}%)"
            for g, n in sorted(genders.items())
        )
        out(f"  gender     : {gender_str}")

        labels = Counter(df["label"])
        out(
            f"  labels     : "
            + ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        )

        methods = Counter(df["spoof_method"])
        out(
            f"  spoof src  : "
            + ", ".join(f"{k}={v}" for k, v in sorted(methods.items()))
        )
        out("")

    # leakage check -- the property that actually matters
    out("=" * 70)
    out("LEAKAGE CHECK")
    out("=" * 70)

    sets = {
        split: set(frames[split]["speaker_id"])
        for split in ("train", "val", "test")
    }

    clean = True
    for a, b in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = sets[a] & sets[b]
        status = "OK" if not overlap else f"LEAK: {sorted(overlap)}"
        if overlap:
            clean = False
        out(f"  {a} vs {b}: {status}")

    out("")
    out(
        "RESULT: speaker-disjoint"
        if clean
        else "RESULT: SPEAKER LEAKAGE DETECTED"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def parse_spoof_root(value: str):
    """Parses 'method=path' into (method, Path)."""
    if "=" not in value:
        print(
            f"ERROR: --spoof-root needs method=path, got: {value}",
            file=sys.stderr,
        )
        sys.exit(1)

    method, path = value.split("=", 1)
    return method.strip(), Path(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build speaker-disjoint, gender-stratified DeepFense "
            "Parquet manifests (70/15/15)."
        )
    )

    parser.add_argument(
        "--bonafide-root",
        required=True,
        help="Processed bonafide audio root",
    )

    parser.add_argument(
        "--spoof-root",
        action="append",
        required=True,
        help=(
            "Spoof set as method=path. Repeatable, e.g. "
            "--spoof-root meta-mms=data\\processed\\meta-mms"
        ),
    )

    parser.add_argument(
        "--speaker-manifest",
        required=True,
        help="manifest_bonafide.csv (for speaker_gender)",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the parquet files",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split (default: 42)",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Use only the first N speakers (smoke testing)",
    )

    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=None,
        help="Use only the first N clips per speaker (smoke testing)",
    )

    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help=(
            "Keep spoof clips that have no bonafide counterpart. Off by "
            "default: the paired design is what rules out the detector "
            "learning content instead of synthesis artefacts."
        ),
    )

    parser.add_argument(
        "--max-unpaired-frac",
        type=float,
        default=0.05,
        help=(
            "Abort if more than this fraction of spoof clips are unpaired "
            "(default 0.05). Exceeding it almost always means the filename "
            "convention changed, not that the data is bad."
        ),
    )

    parser.add_argument(
        "--unpaired-report",
        default=None,
        help="Write the list of excluded clips to this CSV for the record.",
    )

    args = parser.parse_args()

    bonafide_root = Path(args.bonafide_root)
    out_dir = Path(args.out)

    if not bonafide_root.exists():
        print(
            f"ERROR: bonafide root not found: {bonafide_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    speaker_gender = load_speaker_gender(Path(args.speaker_manifest))

    if not speaker_gender:
        print(
            "ERROR: no speaker gender info loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Collect
    # -----------------------------------------------------------------

    print("Collecting bonafide clips...")
    clips_by_speaker = collect_clips(
        bonafide_root,
        LABEL_BONAFIDE,
        "none",
        args.max_clips_per_speaker,
    )

    for entry in args.spoof_root:
        method, root = parse_spoof_root(entry)
        print(f"Collecting spoof clips ({method})...")

        spoof = collect_clips(
            root,
            LABEL_SPOOF,
            method,
            args.max_clips_per_speaker,
        )

        for speaker, rows in spoof.items():
            clips_by_speaker[speaker].extend(rows)

    # -----------------------------------------------------------------
    # Pairing guard
    # -----------------------------------------------------------------

    if args.allow_unpaired:
        print(
            "WARNING: pairing guard disabled (--allow-unpaired). Spoof "
            "clips with no bonafide counterpart will be included, which "
            "reintroduces a content confound.",
            file=sys.stderr,
        )
    else:
        n_spoof_seen, unpaired = enforce_pairing(
            clips_by_speaker, args.max_unpaired_frac
        )
        report_unpaired(
            unpaired,
            n_spoof_seen,
            Path(args.unpaired_report) if args.unpaired_report else None,
        )

    speakers = sorted(s for s in clips_by_speaker if s in speaker_gender)

    dropped = sorted(s for s in clips_by_speaker if s not in speaker_gender)

    if dropped:
        print(
            f"WARNING: {len(dropped)} speaker(s) have audio but no "
            f"gender metadata and were skipped: {dropped[:10]}"
            + (" ..." if len(dropped) > 10 else ""),
            file=sys.stderr,
        )

    if args.max_speakers is not None:
        speakers = speakers[: args.max_speakers]

    if not speakers:
        print("ERROR: no usable speakers found.", file=sys.stderr)
        sys.exit(1)

    print(f"Speakers with audio + metadata: {len(speakers)}")

    # -----------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------

    assignment = split_speakers(speakers, speaker_gender, args.seed)

    rows_by_split = defaultdict(list)

    for speaker in speakers:
        split = assignment[speaker]
        rows_by_split[split].extend(clips_by_speaker[speaker])

    frames = {
        split: pd.DataFrame(rows_by_split[split])
        for split in ("train", "val", "test")
    }

    for split, df in frames.items():
        if df.empty:
            print(
                f"ERROR: {split} split is empty -- too few speakers?",
                file=sys.stderr,
            )
            sys.exit(1)

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    out_dir.mkdir(parents=True, exist_ok=True)

    for split, df in frames.items():
        path = out_dir / f"{split}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(df)} clips)")

    summary = build_summary(frames, assignment, speaker_gender, args.seed)

    summary_path = out_dir / "split_summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Summary written to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
