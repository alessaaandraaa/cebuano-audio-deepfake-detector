r"""
trim_silence.py

Recursively trims leading and trailing silence from WAV files.

The input directory can contain any audio dataset, e.g.:

    data/processed/bonafide/
        0200/
            0200.111020.092714.0001.wav
            ...
        0203/
            ...

    data/processed/meta-mms/
        0200/
            0200.111020.092714.0001.1.wav
            ...

The output directory preserves the exact same folder structure
and filenames.

Only leading and trailing silence are removed.
Silence inside an utterance is preserved.

Original files are NEVER modified.

Examples:

    python preprocessing\trim_silence.py \
        --input data\processed\bonafide \
        --out data\processed\bonafide-trimmed

    python preprocessing\trim_silence.py \
        --input data\processed\meta-mms \
        --out data\processed\meta-mms-trimmed

    python preprocessing\trim_silence.py ^
        --input data\processed\bonafide ^
        --out data\processed\bonafide-trimmed ^
        --threshold -40 ^
        --padding 50
"""

import argparse
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
except ImportError:
    print(
        "Missing dependency: pydub",
        file=sys.stderr,
    )
    print(
        "Install it with:",
        file=sys.stderr,
    )
    print(
        "pip install pydub",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# TRIMMING
# ---------------------------------------------------------------------------


def trim_silence(
    audio: AudioSegment,
    threshold_dbfs: float,
    padding_ms: int,
    min_silence_len_ms: int,
) -> AudioSegment:
    """
    Remove leading and trailing silence.

    Internal pauses are preserved.

    Returns the original audio unchanged if no non-silent
    region can be detected.
    """

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=threshold_dbfs,
    )

    # Entire file appears to be silence.
    if not nonsilent_ranges:
        return audio

    first_start = nonsilent_ranges[0][0]
    last_end = nonsilent_ranges[-1][1]

    # Keep a small amount of padding around speech.
    start = max(
        0,
        first_start - padding_ms,
    )

    end = min(
        len(audio),
        last_end + padding_ms,
    )

    return audio[start:end]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description="Recursively trim leading/trailing silence from WAV files."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing WAV files.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for trimmed WAV files.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-40.0,
        help="Silence threshold in dBFS. Default: -40",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=50,
        help=(
            "Milliseconds of audio to preserve "
            "before/after speech. Default: 50"
        ),
    )

    parser.add_argument(
        "--min-silence",
        type=int,
        default=50,
        help=(
            "Minimum silence duration in milliseconds "
            "used for detection. Default: 50"
        ),
    )

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.out)

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------

    if not input_root.exists():
        print(
            f"ERROR: input directory not found: {input_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not input_root.is_dir():
        print(
            f"ERROR: input path is not a directory: {input_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.padding < 0:
        print(
            "ERROR: --padding cannot be negative.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.min_silence <= 0:
        print(
            "ERROR: --min-silence must be greater than 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Find WAV files
    # -----------------------------------------------------------------------

    wav_files = sorted(input_root.rglob("*.wav"))

    if not wav_files:
        print(
            f"ERROR: no WAV files found under {input_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("Audio Silence Trimming")
    print("=" * 60)

    print(f"Input          : {input_root.resolve()}")

    print(f"Output         : {output_root.resolve()}")

    print(f"Threshold      : {args.threshold} dBFS")

    print(f"Padding        : {args.padding} ms")

    print(f"Min silence    : {args.min_silence} ms")

    print(f"WAV files found: {len(wav_files)}")

    print("-" * 60)

    # -----------------------------------------------------------------------
    # Counters
    # -----------------------------------------------------------------------

    total_processed = 0
    total_trimmed = 0
    total_unchanged = 0
    total_failed = 0

    total_original_ms = 0
    total_output_ms = 0

    # -----------------------------------------------------------------------
    # Process files
    # -----------------------------------------------------------------------

    for index, input_path in enumerate(
        wav_files,
        start=1,
    ):

        # Preserve the exact relative path.
        #
        # Example:
        #
        # input:
        #   bonafide/0200/file.wav
        #
        # output:
        #   bonafide-trimmed/0200/file.wav
        #
        relative_path = input_path.relative_to(input_root)

        output_path = output_root / relative_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            audio = AudioSegment.from_wav(input_path)

            original_duration = len(audio)

            trimmed_audio = trim_silence(
                audio=audio,
                threshold_dbfs=args.threshold,
                padding_ms=args.padding,
                min_silence_len_ms=args.min_silence,
            )

            output_duration = len(trimmed_audio)

            # ---------------------------------------------------------------
            # Export
            # ---------------------------------------------------------------

            trimmed_audio.export(
                output_path,
                format="wav",
            )

            # ---------------------------------------------------------------
            # Statistics
            # ---------------------------------------------------------------

            total_processed += 1

            total_original_ms += original_duration
            total_output_ms += output_duration

            if output_duration < original_duration:
                total_trimmed += 1
            else:
                total_unchanged += 1

            # ---------------------------------------------------------------
            # Progress
            # ---------------------------------------------------------------

            if index % 100 == 0 or index == len(wav_files):

                print(
                    f"{index}/{len(wav_files)} processed "
                    f"({total_trimmed} trimmed, "
                    f"{total_unchanged} unchanged, "
                    f"{total_failed} failed)"
                )

        except Exception as e:

            total_failed += 1

            print(
                f"\nFAILED: {input_path}",
                file=sys.stderr,
            )

            print(
                f"  Error: {e}",
                file=sys.stderr,
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Files processed : {total_processed}")

    print(f"Files trimmed   : {total_trimmed}")

    print(f"Files unchanged : {total_unchanged}")

    print(f"Files failed    : {total_failed}")

    # -----------------------------------------------------------------------
    # Duration statistics
    # -----------------------------------------------------------------------

    if total_original_ms > 0:

        original_seconds = total_original_ms / 1000

        output_seconds = total_output_ms / 1000

        removed_seconds = (total_original_ms - total_output_ms) / 1000

        print()
        print(f"Original duration : {original_seconds / 3600:.2f} hours")

        print(f"Output duration   : {output_seconds / 3600:.2f} hours")

        print(f"Silence removed   : {removed_seconds / 3600:.2f} hours")

    print()
    print(f"Output directory: {output_root.resolve()}")

    if total_failed:
        print()
        print(
            "WARNING: Some files failed. Review the FAILED messages above.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
