"""
generate_mms.py

Generates synthetic Cebuano speech using Meta MMS TTS.

Input:

    mms_selected_speakers.csv

        speaker_id,speaker_gender
        0200,male
        0203,male
        0204,male

    transcripts_mms/
        0200.txt
        0203.txt
        0204.txt

Each transcript file contains:

    filename.wav<TAB>normalized_transcript

Example:

    0200.111020.092714.0121.wav    maayong buntag
    0200.111020.092714.0122.wav    asa ka padulong

IMPORTANT:
    This script performs NO text normalization.

    All transcript normalization required for MMS must already have
    been performed by normalize_mms.py.

Output:

    data/processed/meta-mms/
        0200/
            0200.111020.092714.0121.1.wav
            0200.111020.092714.0122.1.wav

The ".1" suffix identifies MMS-generated speech.
The ".2" suffix is reserved for ElevenLabs-generated speech.

Raw/original audio is never modified.

Usage:

    python preprocessing\\meta_mms\\generate_mms.py --speakers manifests\\mms_selected_speakers.csv --transcripts transcripts_mms --out data\\processed\\meta-mms

Test only 10 utterances per speaker:

    python preprocessing\\generate_mms.py ^
        --speakers manifests\\mms_selected_speakers.csv ^
        --transcripts transcripts_mms ^
        --out data\\processed\\meta-mms ^
        --limit 10
"""

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DEPENDENCIES
# ---------------------------------------------------------------------------

try:
    import torch
    import soundfile as sf
    from transformers import (
        VitsModel,
        VitsTokenizer,
        set_seed,
    )

except ImportError:
    print(
        "Missing dependency(s). Run:",
        file=sys.stderr,
    )

    print(
        "pip install torch transformers soundfile",
        file=sys.stderr,
    )

    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/mms-tts-ceb"

OUTPUT_SAMPLE_RATE = 16000

OUTPUT_SUFFIX = ".1.wav"

SEED = 42

set_seed(SEED)


# ---------------------------------------------------------------------------
# LOAD SPEAKERS
# ---------------------------------------------------------------------------

def load_selected_speakers(
    csv_path: Path,
) -> list[str]:
    """
    Load speaker IDs from the speaker-selection CSV.

    Expected CSV:

        speaker_id,speaker_gender
        0200,male
        0203,male
    """

    speakers = []

    with open(
        csv_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            print(
                "ERROR: speaker CSV has no header.",
                file=sys.stderr,
            )

            sys.exit(1)

        if "speaker_id" not in reader.fieldnames:

            print(
                "ERROR: speaker CSV must contain "
                "'speaker_id' column.",
                file=sys.stderr,
            )

            sys.exit(1)

        for row in reader:

            speaker_id = row[
                "speaker_id"
            ].strip()

            if speaker_id:
                speakers.append(
                    speaker_id
                )

    return speakers


# ---------------------------------------------------------------------------
# LOAD TRANSCRIPT
# ---------------------------------------------------------------------------

def load_transcript_file(
    transcript_path: Path,
):
    """
    Load one speaker's already-normalized transcript file.

    Expected format:

        filename.wav<TAB>normalized transcript

    This function intentionally does NOT modify the transcript text.

    It only:
        - validates the line structure
        - separates filename and transcript
        - removes surrounding whitespace
    """

    entries = []

    with open(
        transcript_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.rstrip(
                "\n\r"
            )

            if not line.strip():
                continue

            # --------------------------------------------------------------
            # TAB delimiter
            # --------------------------------------------------------------

            if "\t" not in line:

                print(
                    f"WARNING: "
                    f"{transcript_path.name}:"
                    f"{line_number}: "
                    f"no TAB delimiter found; "
                    f"skipping.",
                    file=sys.stderr,
                )

                continue

            filename, transcript = (
                line.split(
                    "\t",
                    1,
                )
            )

            filename = filename.strip()
            transcript = transcript.strip()

            # --------------------------------------------------------------
            # Validate filename
            # --------------------------------------------------------------

            if not filename:

                print(
                    f"WARNING: "
                    f"{transcript_path.name}:"
                    f"{line_number}: "
                    f"empty filename; "
                    f"skipping.",
                    file=sys.stderr,
                )

                continue

            if not filename.lower().endswith(
                ".wav"
            ):

                print(
                    f"WARNING: "
                    f"{transcript_path.name}:"
                    f"{line_number}: "
                    f"filename is not a WAV "
                    f"file; skipping.",
                    file=sys.stderr,
                )

                continue

            # --------------------------------------------------------------
            # Validate transcript
            # --------------------------------------------------------------

            if not transcript:

                print(
                    f"WARNING: "
                    f"{transcript_path.name}:"
                    f"{line_number}: "
                    f"empty transcript; "
                    f"skipping.",
                    file=sys.stderr,
                )

                continue

            # --------------------------------------------------------------
            # Keep transcript EXACTLY as provided by normalize_mms.py
            # --------------------------------------------------------------

            entries.append(
                (
                    filename,
                    transcript,
                )
            )

    return entries


# ---------------------------------------------------------------------------
# OUTPUT FILENAME
# ---------------------------------------------------------------------------

def make_output_filename(
    original_filename: str,
) -> str:
    """
    Convert:

        example.wav

    into:

        example.1.wav
    """

    path = Path(
        original_filename
    )

    return (
        f"{path.stem}"
        f"{OUTPUT_SUFFIX}"
    )


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def generate_audio(
    model,
    tokenizer,
    text: str,
    device,
):
    """
    Generate audio from an already-normalized MMS transcript.

    IMPORTANT:
        No normalization happens here.

    The text passed to the tokenizer is exactly the text supplied
    by normalize_mms.py.
    """

    if not text.strip():

        raise ValueError(
            "Transcript is empty."
        )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    inputs = tokenizer(
        text=text,
        return_tensors="pt",
        normalize=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # ------------------------------------------------------------------
    # VITS generation parameters
    # ------------------------------------------------------------------

    model.speaking_rate = 0.95

    model.noise_scale = 0.333

    model.noise_scale_duration = 0.4

    # ------------------------------------------------------------------
    # Generate waveform
    # ------------------------------------------------------------------

    with torch.no_grad():

        output = model(
            **inputs
        ).waveform

    waveform = (
        output
        .squeeze()
        .cpu()
    )

    return waveform


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Cebuano synthetic speech "
            "using Meta MMS TTS."
        )
    )

    parser.add_argument(
        "--speakers",
        required=True,
        help=(
            "CSV containing selected MMS "
            "speakers."
        ),
    )

    parser.add_argument(
        "--transcripts",
        required=True,
        help=(
            "Folder containing one already-"
            "normalized TXT file per speaker."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for MMS-generated "
            "audio."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of utterances to "
            "generate per speaker. Useful for "
            "testing."
        ),
    )

    args = parser.parse_args()

    speakers_path = Path(
        args.speakers
    )

    transcripts_root = Path(
        args.transcripts
    )

    output_root = Path(
        args.out
    )

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------

    if not speakers_path.exists():

        print(
            f"ERROR: speaker CSV not found: "
            f"{speakers_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not transcripts_root.exists():

        print(
            f"ERROR: transcript directory "
            f"not found: "
            f"{transcripts_root}",
            file=sys.stderr,
        )

        sys.exit(1)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Meta MMS Cebuano TTS")
    print("=" * 60)

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Device      : {device}"
    )

    print(
        f"Output rate : "
        f"{OUTPUT_SAMPLE_RATE} Hz"
    )

    print(
        f"Seed        : {SEED}"
    )

    print(
        f"Transcripts : "
        f"{transcripts_root.resolve()}"
    )

    print(
        f"Output      : "
        f"{output_root.resolve()}"
    )

    if args.limit is not None:

        print(
            f"Test limit  : "
            f"{args.limit} utterances/speaker"
        )

    print()

    # -----------------------------------------------------------------------
    # Load model + tokenizer ONCE
    # -----------------------------------------------------------------------

    print(
        f"Loading {MODEL_NAME}..."
    )

    tokenizer = (
        VitsTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        VitsModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(device)

    model.eval()

    print(
        "Model loaded."
    )

    print()

    # -----------------------------------------------------------------------
    # Load speaker selection
    # -----------------------------------------------------------------------

    speakers = load_selected_speakers(
        speakers_path
    )

    print(
        f"Selected speakers: "
        f"{len(speakers)}"
    )

    print("-" * 60)

    # -----------------------------------------------------------------------
    # Counters
    # -----------------------------------------------------------------------

    total_generated = 0

    total_skipped = 0

    total_failed = 0

    total_missing_transcripts = 0

    # -----------------------------------------------------------------------
    # Process speakers
    # -----------------------------------------------------------------------

    for speaker_id in speakers:

        transcript_path = (
            transcripts_root
            / f"{speaker_id}.txt"
        )

        speaker_output_dir = (
            output_root
            / speaker_id
        )

        # --------------------------------------------------------------
        # Missing transcript
        # --------------------------------------------------------------

        if not transcript_path.exists():

            print(
                f"WARNING: [{speaker_id}] "
                f"transcript not found: "
                f"{transcript_path}",
                file=sys.stderr,
            )

            total_missing_transcripts += 1

            continue

        # --------------------------------------------------------------
        # Load transcript
        # --------------------------------------------------------------

        entries = load_transcript_file(
            transcript_path
        )

        if args.limit is not None:

            entries = entries[
                :args.limit
            ]

        print(
            f"[{speaker_id}] "
            f"{len(entries)} utterances"
        )

        # --------------------------------------------------------------
        # Create output directory
        # --------------------------------------------------------------

        speaker_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Generate
        # --------------------------------------------------------------

        for index, (
            original_filename,
            transcript,
        ) in enumerate(
            entries,
            start=1,
        ):

            output_filename = (
                make_output_filename(
                    original_filename
                )
            )

            output_path = (
                speaker_output_dir
                / output_filename
            )

            # ----------------------------------------------------------
            # Resume support
            # ----------------------------------------------------------

            if output_path.exists():

                total_skipped += 1

                continue

            try:

                waveform = generate_audio(
                    model=model,
                    tokenizer=tokenizer,
                    text=transcript,
                    device=device,
                )

                # ------------------------------------------------------
                # Write WAV
                # ------------------------------------------------------

                sf.write(
                    str(output_path),
                    waveform.numpy(),
                    OUTPUT_SAMPLE_RATE,
                )

                total_generated += 1

                # ------------------------------------------------------
                # Show first few examples
                # ------------------------------------------------------

                if index <= 3:

                    print(
                        f"\n  Example {index}:"
                    )

                    print(
                        f"    File       : "
                        f"{original_filename}"
                    )

                    print(
                        f"    MMS input  : "
                        f"{transcript}"
                    )

                    print(
                        f"    Output     : "
                        f"{output_filename}"
                    )

            except Exception as e:

                total_failed += 1

                print(
                    f"\nFAILED "
                    f"[{speaker_id}] "
                    f"{original_filename}"
                )

                print(
                    f"  Transcript: "
                    f"{transcript}"
                )

                print(
                    f"  Error: {e}",
                    file=sys.stderr,
                )

            # ----------------------------------------------------------
            # Progress
            # ----------------------------------------------------------

            if (
                index % 100 == 0
                or index == len(entries)
            ):

                print(
                    f"  {index}/"
                    f"{len(entries)} "
                    f"processed "
                    f"({total_generated} "
                    f"generated, "
                    f"{total_skipped} "
                    f"skipped, "
                    f"{total_failed} "
                    f"failed)"
                )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()

    print("=" * 60)

    print("DONE")

    print("=" * 60)

    print(
        f"Speakers selected       : "
        f"{len(speakers)}"
    )

    print(
        f"Missing transcripts     : "
        f"{total_missing_transcripts}"
    )

    print(
        f"Files generated         : "
        f"{total_generated}"
    )

    print(
        f"Files already existed   : "
        f"{total_skipped}"
    )

    print(
        f"Files failed            : "
        f"{total_failed}"
    )

    print()

    print(
        f"MMS audio written to: "
        f"{output_root.resolve()}"
    )

    if total_failed > 0:

        print(
            "\nWARNING: Some files failed. "
            "Review the FAILED messages above."
        )


if __name__ == "__main__":
    main()