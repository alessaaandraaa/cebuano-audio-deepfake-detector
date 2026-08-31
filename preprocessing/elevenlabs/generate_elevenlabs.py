"""
generate_elevenlabs.py

Generates Cebuano synthetic speech using ElevenLabs cloned voices.

Speaker selection:
    manifests/elevenlabs_selected_speakers.csv

Voice IDs:
    manifests/elevenlabs_voice_ids.txt

Expected voice ID format:

    0201 - VOICE_ID
    0202 - VOICE_ID
    0203 - VOICE_ID

Input transcripts:

    transcripts/
        0201.txt
        0202.txt
        ...

Reference clips:

    elevenlabs-reference/
        0201/
            reference1.wav
            reference2.wav
            reference3.wav

If a transcript WAV filename matches one of the reference
audio filenames for that speaker, that utterance is skipped.

Output:

    data/processed/elevenlabs/
        0201/
            0200.111020.092714.0091.2.wav
            ...
        0202/
            ...

The ".2.wav" suffix identifies ElevenLabs-generated audio.

Model:
    eleven_v3

Language:
    Cebuano (ceb)

Batching:
    --batch 1 = speakers 1-30
    --batch 2 = speakers 31-60
    --batch 3 = speakers 61-90
    ...

The batch is based on the ORDER of speakers in:
    manifests/elevenlabs_selected_speakers.csv

Batch size is fixed at 30.

Examples:

    # Process batch 1
    python preprocessing/elevenlabs/generate_elevenlabs.py --batch 1

    # Process batch 2
    python preprocessing/elevenlabs/generate_elevenlabs.py --batch 2

    # Test first 20 utterances for one speaker
    python preprocessing/elevenlabs/generate_elevenlabs.py \
        --speaker 0202 \
        --limit 20

    # Test first 20 utterances for every speaker in batch 2
    python preprocessing/elevenlabs/generate_elevenlabs.py \
        --batch 2 \
        --limit 20

    # Regenerate existing files
    python preprocessing/elevenlabs/generate_elevenlabs.py \
        --batch 2 \
        --overwrite

Requires:

    pip install requests python-dotenv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRANSCRIPTS_ROOT = (
    PROJECT_ROOT / "transcripts"
)

DEFAULT_REFERENCE_ROOT = (
    PROJECT_ROOT / "elevenlabs-reference"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "elevenlabs"
)

DEFAULT_SPEAKER_MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "elevenlabs_selected_speakers.csv"
)

DEFAULT_VOICE_ID_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "elevenlabs_voice_ids.txt"
)


# ---------------------------------------------------------------------------
# BATCHING
# ---------------------------------------------------------------------------

BATCH_SIZE = 30


# ---------------------------------------------------------------------------
# ELEVENLABS
# ---------------------------------------------------------------------------

API_URL_TEMPLATE = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
)

MODEL_ID = "eleven_v3"

LANGUAGE_CODE = "ceb"

REQUEST_TIMEOUT = 120

REQUEST_DELAY = 0.25


# ---------------------------------------------------------------------------
# SPEAKER MANIFEST
# ---------------------------------------------------------------------------

def load_selected_speakers(
    manifest_path: Path,
) -> list[str]:
    """
    Load speaker IDs from:

        elevenlabs_selected_speakers.csv

    Expected:

        speaker_id,gender
        0201,female
        0202,male

    Only the first column is used.

    The order in this file is IMPORTANT because it determines
    batch membership.
    """

    if not manifest_path.exists():
        print(
            "ERROR: ElevenLabs speaker manifest not found:\n"
            f"  {manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    speakers = []

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.reader(f)

            for line_number, row in enumerate(
                reader,
                start=1,
            ):

                if not row:
                    continue

                speaker_id = row[0].strip()

                if not speaker_id:
                    continue

                # Optional header.
                if (
                    line_number == 1
                    and speaker_id.lower()
                    in {
                        "speaker_id",
                        "speaker",
                        "id",
                    }
                ):
                    continue

                speakers.append(speaker_id)

    except Exception as e:
        print(
            "ERROR: failed to read speaker manifest:\n"
            f"  {manifest_path}\n"
            f"  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Remove duplicates while preserving order.
    speakers = list(
        dict.fromkeys(speakers)
    )

    if not speakers:
        print(
            "ERROR: no speakers found in:\n"
            f"  {manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return speakers


# ---------------------------------------------------------------------------
# BATCH SELECTION
# ---------------------------------------------------------------------------

def select_batch(
    speakers: list[str],
    batch_number: int,
) -> list[str]:
    """
    Select one reproducible batch of speakers.

    Batch size is fixed at 30.

    Example:

        batch 1 -> indices 0-29
        batch 2 -> indices 30-59
        batch 3 -> indices 60-89

    Batch numbering starts at 1.
    """

    if batch_number <= 0:
        print(
            "ERROR: --batch must be 1 or greater.",
            file=sys.stderr,
        )
        sys.exit(1)

    start = (
        (batch_number - 1)
        * BATCH_SIZE
    )

    end = (
        start
        + BATCH_SIZE
    )

    batch = speakers[start:end]

    if not batch:
        total_batches = (
            len(speakers)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        print(
            f"ERROR: batch {batch_number} does not exist.",
            file=sys.stderr,
        )

        print(
            f"Total speakers : {len(speakers)}",
            file=sys.stderr,
        )

        print(
            f"Batch size     : {BATCH_SIZE}",
            file=sys.stderr,
        )

        print(
            f"Available batches: 1-{total_batches}",
            file=sys.stderr,
        )

        sys.exit(1)

    return batch


# ---------------------------------------------------------------------------
# VOICE IDS
# ---------------------------------------------------------------------------

def load_voice_ids(
    voice_id_path: Path,
) -> dict[str, str]:
    """
    Load saved ElevenLabs voice IDs.

    Expected:

        0201 - VOICE_ID
        0202 - VOICE_ID

    Returns:

        {
            "0201": "VOICE_ID",
            "0202": "VOICE_ID",
        }
    """

    if not voice_id_path.exists():
        print(
            "ERROR: voice ID file not found:\n"
            f"  {voice_id_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    voice_ids = {}

    try:
        with open(
            voice_id_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                if " - " not in line:
                    print(
                        "WARNING: ignoring malformed voice ID "
                        f"line {line_number}: {line!r}",
                        file=sys.stderr,
                    )
                    continue

                speaker_id, voice_id = line.split(
                    " - ",
                    1,
                )

                speaker_id = speaker_id.strip()
                voice_id = voice_id.strip()

                if not speaker_id or not voice_id:
                    print(
                        "WARNING: ignoring incomplete voice ID "
                        f"line {line_number}: {line!r}",
                        file=sys.stderr,
                    )
                    continue

                voice_ids[speaker_id] = voice_id

    except Exception as e:
        print(
            "ERROR: failed to read voice ID file:\n"
            f"  {voice_id_path}\n"
            f"  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not voice_ids:
        print(
            "ERROR: no voice IDs found in:\n"
            f"  {voice_id_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return voice_ids


# ---------------------------------------------------------------------------
# REFERENCE CLIPS
# ---------------------------------------------------------------------------

def load_reference_filenames(
    reference_root: Path,
    speaker_id: str,
) -> set[str]:
    """
    Return exact filenames of reference clips for a speaker.

    Example:

        elevenlabs-reference/0202/

    containing:

        0202.abc.wav
        0202.def.wav
        0202.xyz.wav

    returns:

        {
            "0202.abc.wav",
            "0202.def.wav",
            "0202.xyz.wav",
        }
    """

    speaker_dir = (
        reference_root
        / speaker_id
    )

    if not speaker_dir.exists():
        print(
            "WARNING: reference directory not found "
            f"for speaker {speaker_id}:\n"
            f"  {speaker_dir}",
            file=sys.stderr,
        )
        return set()

    if not speaker_dir.is_dir():
        print(
            "WARNING: reference path is not a directory "
            f"for speaker {speaker_id}:\n"
            f"  {speaker_dir}",
            file=sys.stderr,
        )
        return set()

    return {
        path.name
        for path in speaker_dir.iterdir()
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# TRANSCRIPTS
# ---------------------------------------------------------------------------

def load_transcript(
    transcript_path: Path,
) -> list[tuple[str, str]]:
    """
    Load:

        WAV_FILENAME<TAB>TRANSCRIPT

    Returns:

        [
            ("foo.wav", "hello"),
            ("bar.wav", "world"),
        ]
    """

    if not transcript_path.exists():
        print(
            "WARNING: transcript not found:\n"
            f"  {transcript_path}",
            file=sys.stderr,
        )
        return []

    utterances = []

    try:
        with open(
            transcript_path,
            "r",
            encoding="utf-8",
            errors="replace",
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

                if "\t" not in line:
                    print(
                        f"WARNING: {transcript_path.name}:"
                        f"{line_number}: "
                        "no TAB separator; skipping",
                        file=sys.stderr,
                    )
                    continue

                filename, text = line.split(
                    "\t",
                    1,
                )

                filename = filename.strip()
                text = text.strip()

                if not filename:
                    print(
                        f"WARNING: {transcript_path.name}:"
                        f"{line_number}: "
                        "empty filename; skipping",
                        file=sys.stderr,
                    )
                    continue

                if not text:
                    print(
                        f"WARNING: {transcript_path.name}:"
                        f"{line_number}: "
                        "empty transcript; skipping",
                        file=sys.stderr,
                    )
                    continue

                utterances.append(
                    (
                        filename,
                        text,
                    )
                )

    except Exception as e:
        print(
            "ERROR: failed to read transcript:\n"
            f"  {transcript_path}\n"
            f"  {e}",
            file=sys.stderr,
        )

    return utterances


# ---------------------------------------------------------------------------
# ELEVENLABS GENERATION
# ---------------------------------------------------------------------------

def generate_audio(
    api_key: str,
    voice_id: str,
    text: str,
) -> bytes | None:
    """
    Generate speech using ElevenLabs Eleven v3.

    Explicitly requests Cebuano:

        language_code = "ceb"
    """

    url = API_URL_TEMPLATE.format(
        voice_id=voice_id
    )

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }

    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "language_code": LANGUAGE_CODE,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as e:
        print(
            f"  REQUEST ERROR: {e}",
            file=sys.stderr,
        )
        return None

    if response.status_code != 200:
        print(
            f"  API ERROR: HTTP {response.status_code}",
            file=sys.stderr,
        )

        try:
            print(
                response.json(),
                file=sys.stderr,
            )
        except Exception:
            print(
                response.text,
                file=sys.stderr,
            )

        return None

    if not response.content:
        print(
            "  API ERROR: empty audio response",
            file=sys.stderr,
        )
        return None

    return response.content


# ---------------------------------------------------------------------------
# SPEAKER PROCESSING
# ---------------------------------------------------------------------------

def process_speaker(
    speaker_id: str,
    voice_id: str,
    transcripts_root: Path,
    reference_root: Path,
    output_root: Path,
    api_key: str,
    limit: int | None,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    """
    Process one speaker.

    Returns:

        generated,
        skipped_reference,
        skipped_existing,
        failed
    """

    transcript_path = (
        transcripts_root
        / f"{speaker_id}.txt"
    )

    if not transcript_path.exists():
        print(
            f"\nWARNING: no transcript for speaker "
            f"{speaker_id}:\n"
            f"  {transcript_path}",
            file=sys.stderr,
        )

        return 0, 0, 0, 0

    reference_filenames = (
        load_reference_filenames(
            reference_root,
            speaker_id,
        )
    )

    utterances = load_transcript(
        transcript_path
    )

    if limit is not None:
        utterances = utterances[:limit]

    # IMPORTANT:
    # Each speaker gets their own output folder.
    speaker_output_root = (
        output_root
        / speaker_id
    )

    speaker_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(f"SPEAKER: {speaker_id}")
    print("=" * 70)

    print(
        f"Voice ID          : {voice_id}"
    )

    print(
        f"Transcript        : "
        f"{transcript_path}"
    )

    print(
        f"Output folder     : "
        f"{speaker_output_root}"
    )

    print(
        f"Reference clips   : "
        f"{len(reference_filenames)}"
    )

    print(
        f"Utterances loaded : "
        f"{len(utterances)}"
    )

    if limit is not None:
        print(
            f"Test limit        : "
            f"{limit}"
        )

    if reference_filenames:
        print()
        print(
            "Reference files that will be skipped:"
        )

        for filename in sorted(
            reference_filenames
        ):
            print(
                f"  - {filename}"
            )

    print("-" * 70)

    generated = 0
    skipped_reference = 0
    skipped_existing = 0
    failed = 0

    for index, (
        wav_filename,
        text,
    ) in enumerate(
        utterances,
        start=1,
    ):

        # --------------------------------------------------------------
        # Skip reference audio
        # --------------------------------------------------------------

        if wav_filename in reference_filenames:

            skipped_reference += 1

            print(
                f"[{index}/{len(utterances)}] "
                f"SKIP REFERENCE: "
                f"{wav_filename}"
            )

            continue

        # --------------------------------------------------------------
        # Output name
        #
        # Input:
        #
        #   0200.111020.092714.0091.wav
        #
        # Output:
        #
        #   0200.111020.092714.0091.2.wav
        # --------------------------------------------------------------

        input_stem = Path(
            wav_filename
        ).stem

        output_filename = (
            f"{input_stem}.2.wav"
        )

        output_path = (
            speaker_output_root
            / output_filename
        )

        # --------------------------------------------------------------
        # Skip existing
        # --------------------------------------------------------------

        if (
            output_path.exists()
            and not overwrite
        ):

            skipped_existing += 1

            print(
                f"[{index}/{len(utterances)}] "
                f"EXISTS: "
                f"{output_path.name}"
            )

            continue

        # --------------------------------------------------------------
        # Generate
        # --------------------------------------------------------------

        print(
            f"[{index}/{len(utterances)}] "
            f"GENERATING: "
            f"{wav_filename}"
        )

        print(
            f"    Text: {text}"
        )

        audio = generate_audio(
            api_key=api_key,
            voice_id=voice_id,
            text=text,
        )

        if audio is None:
            failed += 1

            print(
                f"    FAILED: {wav_filename}",
                file=sys.stderr,
            )

            continue

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        try:
            with open(
                output_path,
                "wb",
            ) as f:

                f.write(audio)

        except OSError as e:
            failed += 1

            print(
                f"    SAVE ERROR: "
                f"{output_path}: {e}",
                file=sys.stderr,
            )

            continue

        generated += 1

        print(
            f"    SAVED: "
            f"{output_path}"
        )

        if REQUEST_DELAY > 0:
            time.sleep(
                REQUEST_DELAY
            )

    print("-" * 70)

    print(
        f"[{speaker_id}] "
        f"generated={generated}, "
        f"reference_skipped={skipped_reference}, "
        f"existing_skipped={skipped_existing}, "
        f"failed={failed}"
    )

    return (
        generated,
        skipped_reference,
        skipped_existing,
        failed,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Cebuano ElevenLabs speech "
            "for selected cloned speakers."
        )
    )

    parser.add_argument(
        "--transcripts",
        default=str(
            DEFAULT_TRANSCRIPTS_ROOT
        ),
        help=(
            "Folder containing speaker transcript "
            "files."
        ),
    )

    parser.add_argument(
        "--reference-root",
        default=str(
            DEFAULT_REFERENCE_ROOT
        ),
        help=(
            "Root folder containing per-speaker "
            "ElevenLabs reference clips."
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
        help=(
            "Output root. Each speaker gets "
            "their own subfolder."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_SPEAKER_MANIFEST
        ),
        help=(
            "ElevenLabs selected-speaker CSV."
        ),
    )

    parser.add_argument(
        "--voice-ids",
        default=str(
            DEFAULT_VOICE_ID_FILE
        ),
        help=(
            "File containing speaker -> ElevenLabs "
            "voice ID mappings."
        ),
    )

    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "Process only one speaker. "
            "Example: --speaker 0202"
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "Process one 30-speaker batch. "
            "Batch 1 = speakers 1-30, "
            "batch 2 = speakers 31-60, etc."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N transcript "
            "utterances per selected speaker. "
            "Example: --limit 20"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate audio even when the "
            "corresponding .2.wav file already exists."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate mutually exclusive selection modes
    # ------------------------------------------------------------------

    if (
        args.speaker is not None
        and args.batch is not None
    ):
        print(
            "ERROR: use either --speaker OR --batch, "
            "not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate batch
    # ------------------------------------------------------------------

    if (
        args.batch is not None
        and args.batch <= 0
    ):
        print(
            "ERROR: --batch must be 1 or greater.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate limit
    # ------------------------------------------------------------------

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        print(
            "ERROR: --limit must be greater than 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load environment
    # ------------------------------------------------------------------

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.environ.get(
        "ELEVENLABS_API_KEY"
    )

    if not api_key:
        print(
            "ERROR: ELEVENLABS_API_KEY not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------

    transcripts_root = Path(
        args.transcripts
    )

    reference_root = Path(
        args.reference_root
    )

    output_root = Path(
        args.out
    )

    manifest_path = Path(
        args.manifest
    )

    voice_id_path = Path(
        args.voice_ids
    )

    # ------------------------------------------------------------------
    # Load all speakers
    # ------------------------------------------------------------------

    all_speakers = (
        load_selected_speakers(
            manifest_path
        )
    )

    # ------------------------------------------------------------------
    # Select speakers
    # ------------------------------------------------------------------

    if args.speaker:

        if args.speaker not in all_speakers:
            print(
                f"ERROR: speaker {args.speaker} "
                "is not present in:\n"
                f"  {manifest_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        selected_speakers = [
            args.speaker
        ]

        selection_description = (
            f"single speaker {args.speaker}"
        )

    elif args.batch is not None:

        selected_speakers = select_batch(
            all_speakers,
            args.batch,
        )

        start_number = (
            (args.batch - 1)
            * BATCH_SIZE
            + 1
        )

        end_number = (
            start_number
            + len(selected_speakers)
            - 1
        )

        selection_description = (
            f"batch {args.batch} "
            f"(manifest speakers "
            f"{start_number}-{end_number})"
        )

    else:

        selected_speakers = all_speakers

        selection_description = (
            "all speakers"
        )

    # ------------------------------------------------------------------
    # Load voice IDs
    # ------------------------------------------------------------------

    voice_ids = load_voice_ids(
        voice_id_path
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Missing voice IDs are NOT fatal.
    #
    # This is what allows batching to work when, for example,
    # batch 2 contains speakers whose clones haven't been created yet.
    # ------------------------------------------------------------------

    speakers_with_voice_ids = [
        speaker
        for speaker in selected_speakers
        if speaker in voice_ids
    ]

    speakers_without_voice_ids = [
        speaker
        for speaker in selected_speakers
        if speaker not in voice_ids
    ]

    if speakers_without_voice_ids:

        print()
        print(
            "WARNING: the following selected speakers "
            "do not have saved ElevenLabs voice IDs "
            "and will be skipped:"
        )

        for speaker in speakers_without_voice_ids:
            print(
                f"  - {speaker}"
            )

        print()

    if not speakers_with_voice_ids:

        print(
            "ERROR: none of the selected speakers "
            "have saved ElevenLabs voice IDs.",
            file=sys.stderr,
        )

        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ELEVENLABS CEBUANO GENERATION")
    print("=" * 70)

    print(
        f"Model             : {MODEL_ID}"
    )

    print(
        f"Language code     : {LANGUAGE_CODE}"
    )

    print(
        f"Batch size        : {BATCH_SIZE}"
    )

    print(
        f"Selection         : "
        f"{selection_description}"
    )

    print(
        f"Selected speakers : "
        f"{len(selected_speakers)}"
    )

    print(
        f"With voice IDs    : "
        f"{len(speakers_with_voice_ids)}"
    )

    print(
        f"Missing voice IDs : "
        f"{len(speakers_without_voice_ids)}"
    )

    print(
        f"Transcript root   : "
        f"{transcripts_root.resolve()}"
    )

    print(
        f"Reference root    : "
        f"{reference_root.resolve()}"
    )

    print(
        f"Output root       : "
        f"{output_root.resolve()}"
    )

    print(
        f"Speaker manifest  : "
        f"{manifest_path.resolve()}"
    )

    print(
        f"Voice ID file     : "
        f"{voice_id_path.resolve()}"
    )

    if args.limit is not None:
        print(
            f"Limit             : "
            f"first {args.limit} per speaker"
        )
    else:
        print(
            "Limit             : none"
        )

    print(
        f"Overwrite         : "
        f"{args.overwrite}"
    )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Process speakers
    # ------------------------------------------------------------------

    total_generated = 0
    total_reference_skipped = 0
    total_existing_skipped = 0
    total_failed = 0

    speakers_processed = 0

    for speaker_id in speakers_with_voice_ids:

        voice_id = voice_ids[
            speaker_id
        ]

        (
            generated,
            skipped_reference,
            skipped_existing,
            failed,
        ) = process_speaker(
            speaker_id=speaker_id,
            voice_id=voice_id,
            transcripts_root=transcripts_root,
            reference_root=reference_root,
            output_root=output_root,
            api_key=api_key,
            limit=args.limit,
            overwrite=args.overwrite,
        )

        total_generated += generated

        total_reference_skipped += (
            skipped_reference
        )

        total_existing_skipped += (
            skipped_existing
        )

        total_failed += failed

        speakers_processed += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Selection                : "
        f"{selection_description}"
    )

    print(
        f"Speakers selected        : "
        f"{len(selected_speakers)}"
    )

    print(
        f"Speakers processed       : "
        f"{speakers_processed}"
    )

    print(
        f"Missing voice IDs        : "
        f"{len(speakers_without_voice_ids)}"
    )

    print(
        f"Audio generated          : "
        f"{total_generated}"
    )

    print(
        f"Reference clips skipped  : "
        f"{total_reference_skipped}"
    )

    print(
        f"Existing files skipped   : "
        f"{total_existing_skipped}"
    )

    print(
        f"Generation failures      : "
        f"{total_failed}"
    )

    print(
        f"Output folder            : "
        f"{output_root.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()