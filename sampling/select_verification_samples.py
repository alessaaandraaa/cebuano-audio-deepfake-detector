r"""
select_verification_samples.py

Draws the stratified verification sample used for expert linguistic
review of the synthetic (spoof) portion of the dataset.

Selects 50 ElevenLabs clips and 50 Meta-MMS clips (100 total),
stratified by utterance duration so that short utterances -- where
Meta-MMS is weakest -- are represented rather than washed out by the
corpus-wide duration distribution.

Duration bins:

    short    < 2.0s
    medium   2.0s - 5.0s
    long     > 5.0s

Target allocation per source:

    25% short
    50% medium
    25% long

If a bin has fewer files than its quota, the unused quota is
redistributed to the bins that still have capacity.

Input:

    data/processed/elevenlabs/
    data/processed/meta-mms/
    transcripts/

Output:

    data/samples/
        ceb_sample_001.wav
        ceb_sample_002.wav
        ...
        samples.tsv

The manifest (samples.tsv) records, per sampled clip:

    sample_filename, source, speaker_id, original_filename,
    duration_seconds, duration_bin, transcript

Samples are shuffled before numbering so that ceb_sample_001-100
are not grouped by source.

IMPORTANT:
    Source audio is COPIED, never moved or modified.

    Pass --seed for a reproducible draw. Record the seed used,
    since the sample is the basis for the validation results.

Requires:

    pip install soundfile

Usage:

    python sampling\select_verification_samples.py --seed 42

    python sampling\select_verification_samples.py \
        --elevenlabs-root data\processed\elevenlabs \
        --meta-mms-root data\processed\meta-mms \
        --transcripts-root transcripts \
        --out data\samples \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import soundfile as sf

SHORT_MAX = 2.0
MEDIUM_MAX = 5.0

SHORT_RATIO = 0.25
MEDIUM_RATIO = 0.50
LONG_RATIO = 0.25


def get_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def get_duration_bin(duration: float) -> str:
    if duration < SHORT_MAX:
        return "short"
    elif duration <= MEDIUM_MAX:
        return "medium"
    return "long"


def collect_samples(root: Path):
    samples = []

    # Recursively collect ALL WAVs, regardless of speaker folder.
    for path in root.rglob("*.wav"):
        try:
            duration = get_duration(path)
        except Exception as e:
            print(f"Skipping unreadable file: {path} ({e})")
            continue

        samples.append(
            {
                "path": path,
                "duration": duration,
                "bin": get_duration_bin(duration),
            }
        )

    return samples


def allocate_counts(total: int, bins: dict):
    """
    Allocate approximately:
        25% short
        50% medium
        25% long

    If a duration bin doesn't have enough files,
    redistribute its unused quota.
    """

    short_count = round(total * SHORT_RATIO)
    medium_count = round(total * MEDIUM_RATIO)
    long_count = total - short_count - medium_count

    desired = {
        "short": short_count,
        "medium": medium_count,
        "long": long_count,
    }

    selected_counts = {
        name: min(desired[name], len(bins[name])) for name in bins
    }

    remaining = total - sum(selected_counts.values())

    while remaining > 0:
        candidates = [
            name
            for name in ("short", "medium", "long")
            if selected_counts[name] < len(bins[name])
        ]

        if not candidates:
            break

        candidates.sort(
            key=lambda name: len(bins[name]) - selected_counts[name],
            reverse=True,
        )

        selected_counts[candidates[0]] += 1
        remaining -= 1

    return selected_counts


def stratified_sample(samples, count):
    bins = {
        "short": [],
        "medium": [],
        "long": [],
    }

    for sample in samples:
        bins[sample["bin"]].append(sample)

    counts = allocate_counts(count, bins)

    selected = []

    for bin_name in ("short", "medium", "long"):
        random.shuffle(bins[bin_name])
        selected.extend(bins[bin_name][: counts[bin_name]])

    return selected


def get_speaker_id(wav_path: Path) -> str:
    """
    Example:
        0201.111024.022631.0430.wav
        -> 0201
    """
    return wav_path.stem.split(".")[0]


def find_transcript(
    wav_path: Path,
    transcripts_root: Path,
) -> str:
    """
    Generated audio filenames have a source filename plus
    a generation suffix:

        original.wav.1.wav  -> Meta-MMS
        original.wav.2.wav  -> ElevenLabs

    The transcript files contain the original filename, so
    remove the final .1/.2 suffix before searching.

    Example:

        0201.111024.022631.0430.2.wav
        -> 0201.111024.022631.0430.wav

        transcripts/0201.txt
    """

    speaker_id = get_speaker_id(wav_path)

    transcript_file = transcripts_root / f"{speaker_id}.txt"

    if not transcript_file.exists():
        print(f"WARNING: transcript file not found: {transcript_file}")
        return ""

    # Remove the final generation suffix.
    #
    # 0201....0430.2.wav -> 0201....0430.wav
    # 0201....0430.1.wav -> 0201....0430.wav
    filename = wav_path.name

    if filename.endswith(".1.wav") or filename.endswith(".2.wav"):
        original_filename = filename[:-6] + ".wav"
    else:
        original_filename = filename

    try:
        text = transcript_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as e:
        print(f"WARNING: could not read {transcript_file}: {e}")
        return ""

    for line in text.splitlines():
        if original_filename in line:
            return line.strip()

    print(
        "WARNING: transcript not found for "
        f"{original_filename} in {transcript_file}"
    )

    return ""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample and COPY Cebuano audio "
            "from ElevenLabs and Meta-MMS."
        )
    )

    parser.add_argument(
        "--elevenlabs-root",
        type=Path,
        default=Path("data/processed/elevenlabs"),
    )

    parser.add_argument(
        "--meta-mms-root",
        type=Path,
        default=Path("data/processed/meta-mms"),
    )

    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("transcripts"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/samples"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)

    print("Collecting ElevenLabs samples...")
    elevenlabs = collect_samples(args.elevenlabs_root)

    print("Collecting Meta-MMS samples...")
    meta_mms = collect_samples(args.meta_mms_root)

    print(f"ElevenLabs files found: {len(elevenlabs)}")
    print(f"Meta-MMS files found:   {len(meta_mms)}")

    if len(elevenlabs) < 50:
        raise RuntimeError(
            f"Need at least 50 ElevenLabs files, found {len(elevenlabs)}"
        )

    if len(meta_mms) < 50:
        raise RuntimeError(
            f"Need at least 50 Meta-MMS files, found {len(meta_mms)}"
        )

    selected_elevenlabs = stratified_sample(elevenlabs, 50)
    selected_meta_mms = stratified_sample(meta_mms, 50)

    selected = [("elevenlabs", sample) for sample in selected_elevenlabs] + [
        ("meta-mms", sample) for sample in selected_meta_mms
    ]

    # Shuffle so 001-100 aren't grouped by source.
    random.shuffle(selected)

    manifest_path = args.out / "samples.tsv"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(
            "sample_filename\t"
            "source\t"
            "speaker_id\t"
            "original_filename\t"
            "duration_seconds\t"
            "duration_bin\t"
            "transcript\n"
        )

        for index, (source, sample) in enumerate(
            selected,
            start=1,
        ):
            output_name = f"ceb_sample_{index:03d}.wav"
            output_path = args.out / output_name

            # COPY — the original file is untouched.
            shutil.copy2(
                sample["path"],
                output_path,
            )

            speaker_id = get_speaker_id(sample["path"])

            transcript = find_transcript(
                sample["path"],
                args.transcripts_root,
            )

            # Make sure tabs/newlines in the transcript don't
            # destroy the TSV structure.
            transcript = (
                transcript.replace("\t", " ")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

            manifest.write(
                f"{output_name}\t"
                f"{source}\t"
                f"{speaker_id}\t"
                f"{sample['path'].name}\t"
                f"{sample['duration']:.3f}\t"
                f"{sample['bin']}\t"
                f"{transcript}\n"
            )

            print(
                f"[{index:03d}/100] "
                f"{source:11s} "
                f"{speaker_id} "
                f"{sample['duration']:6.2f}s "
                f"{sample['bin']:6s} -> "
                f"{output_name}"
            )

            print(f"             Transcript: {transcript}")

    print()
    print("Done.")
    print(f"Samples:  {args.out}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Original files were COPIED, not moved.")


if __name__ == "__main__":
    main()
