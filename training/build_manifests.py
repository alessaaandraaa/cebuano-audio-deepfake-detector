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
