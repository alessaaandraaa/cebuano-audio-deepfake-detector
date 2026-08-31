"""
clone_voice.py

Creates ElevenLabs Instant Voice Clones (IVC) for speakers selected
from:

    manifests/elevenlabs_selected_speakers.csv

Reference audio:

    elevenlabs-reference/<speaker_id>/

Voice IDs:

    manifests/elevenlabs_voice_ids.txt

Expected voice ID format:

    0201 - VOICE_ID
    0202 - VOICE_ID
    0203 - VOICE_ID

BATCHING
--------

Speakers are selected reproducibly from their order in the manifest.

Default batch size:

    30 speakers

Examples:

    Batch 1:
        --batch 1

    Batch 2:
        --batch 2

    Batch 3:
        --batch 3

The original CSV is NEVER modified.

If a speaker already has a voice ID in elevenlabs_voice_ids.txt,
that speaker is skipped safely.

This means the batch number determines WHICH speakers belong to
the batch, while the existing voice-ID log determines whether
a speaker still needs to be processed.

Requires:

    pip install requests python-dotenv soundfile

Requires a .env file at the project root containing:

    ELEVENLABS_API_KEY=your_key_here

Usage:

    # Process speakers 1-30
    python preprocessing/elevenlabs/clone_voice.py --batch 1

    # Process speakers 31-60
    python preprocessing/elevenlabs/clone_voice.py --batch 2

    # Process speakers 61-90
    python preprocessing/elevenlabs/clone_voice.py --batch 3

    # Use a different batch size
    python preprocessing/elevenlabs/clone_voice.py --batch 2 --batch-size 20

    # Process one specific speaker
    python preprocessing/elevenlabs/clone_voice.py --speaker 0231

    # Test one speaker without affecting batching
    python preprocessing/elevenlabs/clone_voice.py --speaker 0231

    # Enable background-noise removal
    python preprocessing/elevenlabs/clone_voice.py \
        --batch 2 \
        --remove-background-noise
"""

import argparse
import csv
import json
import mimetypes
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------------

try:
    import soundfile as sf
except ImportError:
    print(
        "Missing dependency: soundfile\n"
        "Run: pip install soundfile",
        file=sys.stderr,
    )
    sys.exit(1)


try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print(
        "Missing dependency(s).\n"
        "Run: pip install requests python-dotenv",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

# clone_voice.py:
#
# preprocessing/
#   elevenlabs/
#     clone_voice.py
#
# parents[0] = elevenlabs
# parents[1] = preprocessing
# parents[2] = project root
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "elevenlabs_selected_speakers.csv"
)


DEFAULT_VOICE_LOG_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "elevenlabs_voice_ids.txt"
)


DEFAULT_REFERENCE_ROOT = (
    PROJECT_ROOT
    / "elevenlabs-reference"
)


# ---------------------------------------------------------------------------
# ELEVENLABS API
# ---------------------------------------------------------------------------

API_URL = (
    "https://api.elevenlabs.io/v1/voices/add"
)


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}


CONTENT_TYPE_MAP = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


# ---------------------------------------------------------------------------
# SPEAKER MANIFEST
# ---------------------------------------------------------------------------

def load_all_selected_speakers(
    manifest_path: Path,
) -> list[str]:
    """
    Load ALL speaker IDs from the selected-speaker manifest.

    Expected format:

        speaker_id,gender
        0201,female
        0202,male

    Only the first column is used.

    IMPORTANT:
        The order in the CSV is preserved.

    This order is what makes batching reproducible.
    """

    if not manifest_path.exists():
        print(
            f"ERROR: ElevenLabs speaker manifest not found:\n"
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

                speakers.append(
                    speaker_id
                )

    except Exception as e:
        print(
            f"ERROR: failed to read speaker manifest:\n"
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
            f"ERROR: no speakers found in:\n"
            f"  {manifest_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return speakers


def select_batch(
    speakers: list[str],
    batch_number: int,
    batch_size: int,
) -> list[str]:
    """
    Select one reproducible batch from the manifest.

    Batch numbering is 1-based.

    Example with batch_size=30:

        batch 1 -> indexes 0-29
        batch 2 -> indexes 30-59
        batch 3 -> indexes 60-89

    The manifest itself is never modified.
    """

    start = (
        batch_number - 1
    ) * batch_size

    end = start + batch_size

    return speakers[start:end]


# ---------------------------------------------------------------------------
# EXISTING VOICE IDS
# ---------------------------------------------------------------------------

def load_existing_voice_ids(
    voice_log_path: Path,
) -> dict[str, str]:
    """
    Load already-created voice IDs.

    Expected format:

        0201 - VOICE_ID
        0202 - VOICE_ID

    Returns:

        {
            "0201": "VOICE_ID",
            "0202": "VOICE_ID",
        }

    Invalid lines are ignored with a warning.
    """

    if not voice_log_path.exists():
        return {}

    existing = {}

    try:
        with open(
            voice_log_path,
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
                        f"WARNING: ignoring malformed voice-log "
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
                        f"WARNING: ignoring malformed voice-log "
                        f"line {line_number}: {line!r}",
                        file=sys.stderr,
                    )
                    continue

                existing[speaker_id] = voice_id

    except Exception as e:
        print(
            f"ERROR: failed to read voice ID log:\n"
            f"  {voice_log_path}\n"
            f"  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    return existing


# ---------------------------------------------------------------------------
# VOICE ID LOGGING
# ---------------------------------------------------------------------------

def append_voice_id(
    voice_log_path: Path,
    speaker_id: str,
    voice_id: str,
):
    """
    Append:

        speaker_id - voice_id

    to the voice log.
    """

    voice_log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        voice_log_path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{speaker_id} - {voice_id}\n"
        )


# ---------------------------------------------------------------------------
# AUDIO VALIDATION
# ---------------------------------------------------------------------------

def validate_audio_file(
    path: Path,
) -> bool:
    """
    Validate that an audio file is readable and contains
    non-empty audio.

    Returns True if valid.
    """

    try:
        info = sf.info(
            str(path)
        )

        duration = (
            info.frames
            / info.samplerate
        )

        if (
            info.frames <= 0
            or info.samplerate <= 0
            or info.channels <= 0
            or duration <= 0
        ):

            print(
                f"  INVALID: {path.name} "
                f"(empty or malformed audio)"
            )

            return False

        print(
            f"  OK      : {path.name} | "
            f"{info.samplerate} Hz | "
            f"{info.channels} ch | "
            f"{duration:.2f}s | "
            f"{info.format} | "
            f"subtype={info.subtype}"
        )

        return True

    except Exception as e:

        print(
            f"  INVALID : {path.name} | {e}",
            file=sys.stderr,
        )

        return False


# ---------------------------------------------------------------------------
# CREATE VOICE
# ---------------------------------------------------------------------------

def create_voice(
    speaker_id: str,
    clip_paths: list[Path],
    api_key: str,
    description: str | None = None,
    remove_background_noise: bool = False,
) -> str | None:
    """
    Create an ElevenLabs Instant Voice Clone.

    Returns:
        voice_id on success
        None on failure
    """

    voice_name = (
        f"ceb-{speaker_id}"
    )

    labels_dict = {
        "language": "ceb",
        "speaker_id": speaker_id,
    }

    headers = {
        "xi-api-key": api_key,
    }

    data = {
        "name": voice_name,
        "labels": json.dumps(
            labels_dict
        ),
        "remove_background_noise": str(
            remove_background_noise
        ).lower(),
    }

    if description:
        data["description"] = description

    open_file_handles = []
    files_payload = []

    try:

        for path in clip_paths:

            content_type = (
                CONTENT_TYPE_MAP.get(
                    path.suffix.lower()
                )
            )

            if content_type is None:

                content_type = (
                    mimetypes.guess_type(
                        str(path)
                    )[0]
                    or "application/octet-stream"
                )

            fh = open(
                path,
                "rb",
            )

            open_file_handles.append(
                fh
            )

            files_payload.append(
                (
                    "files",
                    (
                        path.name,
                        fh,
                        content_type,
                    ),
                )
            )

        print()
        print("=" * 60)
        print(
            f"Uploading {len(clip_paths)} clip(s)"
        )
        print(
            f"Creating voice: {voice_name}"
        )
        print("=" * 60)

        response = requests.post(
            API_URL,
            headers=headers,
            data=data,
            files=files_payload,
            timeout=120,
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("REQUEST ERROR")
        print("=" * 60)
        print(
            f"Speaker: {speaker_id}"
        )
        print(
            f"Error  : {e}"
        )

        return None

    finally:

        for fh in open_file_handles:
            fh.close()

    # ------------------------------------------------------------------
    # API ERROR
    # ------------------------------------------------------------------

    if response.status_code != 200:

        print()
        print("=" * 60)
        print("ElevenLabs API ERROR")
        print("=" * 60)

        print(
            f"Speaker: {speaker_id}"
        )

        print(
            f"Status : {response.status_code}"
        )

        try:
            print(
                json.dumps(
                    response.json(),
                    indent=2,
                )
            )

        except Exception:
            print(
                response.text
            )

        print()
        print(
            "Files that were uploaded:"
        )

        for path in clip_paths:
            print(
                f"  - {path.name}"
            )

        return None

    # ------------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------------

    try:

        result = response.json()

    except Exception as e:

        print(
            f"ERROR: ElevenLabs returned an invalid "
            f"JSON response for {speaker_id}: {e}",
            file=sys.stderr,
        )

        return None

    voice_id = result.get(
        "voice_id"
    )

    if not voice_id:

        print(
            f"ERROR: ElevenLabs response did not "
            f"contain a voice_id for {speaker_id}.",
            file=sys.stderr,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        return None

    return voice_id


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create ElevenLabs IVC voices for "
            "selected Cebuano speakers using "
            "reproducible batches."
        )
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
        help=(
            "CSV containing selected "
            "ElevenLabs speakers."
        ),
    )

    parser.add_argument(
        "--reference-root",
        default=str(
            DEFAULT_REFERENCE_ROOT
        ),
        help=(
            "Root folder containing per-speaker "
            "reference audio folders."
        ),
    )

    parser.add_argument(
        "--voice-log",
        default=str(
            DEFAULT_VOICE_LOG_PATH
        ),
        help=(
            "File containing speaker -> voice ID "
            "mappings."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "1-based batch number. "
            "Example: --batch 2 selects speakers "
            "31-60 when batch size is 30."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help=(
            "Number of speakers per batch. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "Process exactly one speaker instead "
            "of using batching. "
            "Example: --speaker 0231"
        ),
    )

    parser.add_argument(
        "--remove-background-noise",
        action="store_true",
        help=(
            "Ask ElevenLabs to remove background "
            "noise from the uploaded references."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------

    if args.batch is not None and args.batch <= 0:

        print(
            "ERROR: --batch must be greater than 0.",
            file=sys.stderr,
        )

        sys.exit(1)

    if args.batch_size <= 0:

        print(
            "ERROR: --batch-size must be greater than 0.",
            file=sys.stderr,
        )

        sys.exit(1)

    if (
        args.batch is not None
        and args.speaker is not None
    ):

        print(
            "ERROR: use either --batch or --speaker, "
            "not both.",
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
            "ERROR: ELEVENLABS_API_KEY not found.\n"
            "Check your .env file.",
            file=sys.stderr,
        )

        sys.exit(1)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    manifest_path = Path(
        args.manifest
    )

    reference_root = Path(
        args.reference_root
    )

    voice_log_path = Path(
        args.voice_log
    )

    # ------------------------------------------------------------------
    # Load complete manifest
    # ------------------------------------------------------------------

    all_speakers = (
        load_all_selected_speakers(
            manifest_path
        )
    )

    # ------------------------------------------------------------------
    # Determine speakers to process
    # ------------------------------------------------------------------

    if args.speaker:

        if args.speaker not in all_speakers:

            print(
                f"ERROR: speaker {args.speaker} "
                f"is not present in:\n"
                f"  {manifest_path}",
                file=sys.stderr,
            )

            sys.exit(1)

        selected_speakers = [
            args.speaker
        ]

        batch_description = (
            f"single speaker: {args.speaker}"
        )

    else:

        # --------------------------------------------------------------
        # Explicit batch mode
        # --------------------------------------------------------------

        if args.batch is None:

            print(
                "ERROR: specify either:\n"
                "  --batch N\n"
                "or:\n"
                "  --speaker SPEAKER_ID",
                file=sys.stderr,
            )

            sys.exit(1)

        selected_speakers = select_batch(
            speakers=all_speakers,
            batch_number=args.batch,
            batch_size=args.batch_size,
        )

        if not selected_speakers:

            start_number = (
                (args.batch - 1)
                * args.batch_size
                + 1
            )

            print(
                f"ERROR: batch {args.batch} contains "
                f"no speakers.",
                file=sys.stderr,
            )

            print(
                f"Requested range: "
                f"{start_number}-"
                f"{start_number + args.batch_size - 1}",
                file=sys.stderr,
            )

            print(
                f"Manifest contains only "
                f"{len(all_speakers)} speakers.",
                file=sys.stderr,
            )

            sys.exit(1)

        start_index = (
            (args.batch - 1)
            * args.batch_size
        )

        end_index = (
            start_index
            + len(selected_speakers)
        )

        batch_description = (
            f"batch {args.batch} "
            f"(manifest positions "
            f"{start_index + 1}-{end_index})"
        )

    # ------------------------------------------------------------------
    # Load already-created voices
    # ------------------------------------------------------------------

    existing_voice_ids = (
        load_existing_voice_ids(
            voice_log_path
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ELEVENLABS INSTANT VOICE CLONE BATCH"
    )
    print("=" * 70)

    print(
        f"Manifest       : "
        f"{manifest_path.resolve()}"
    )

    print(
        f"Reference root : "
        f"{reference_root.resolve()}"
    )

    print(
        f"Voice ID log   : "
        f"{voice_log_path.resolve()}"
    )

    print(
        f"Total speakers : "
        f"{len(all_speakers)}"
    )

    print(
        f"Selection      : "
        f"{batch_description}"
    )

    if not args.speaker:

        print(
            f"Batch size     : "
            f"{args.batch_size}"
        )

    print(
        f"Speakers       : "
        f"{len(selected_speakers)}"
    )

    print(
        f"Already cloned : "
        f"{len(existing_voice_ids)}"
    )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Show selected speakers
    # ------------------------------------------------------------------

    print()
    print(
        "Speakers selected for this run:"
    )

    for position, speaker_id in enumerate(
        selected_speakers,
        start=1,
    ):

        if speaker_id in existing_voice_ids:

            print(
                f"  {position:>2}. "
                f"{speaker_id} "
                f"[ALREADY DONE]"
            )

        else:

            print(
                f"  {position:>2}. "
                f"{speaker_id}"
            )

    print("-" * 70)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    created_count = 0
    skipped_count = 0
    failed_count = 0

    # ------------------------------------------------------------------
    # Process speakers
    # ------------------------------------------------------------------

    for index, speaker_id in enumerate(
        selected_speakers,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"[{index}/{len(selected_speakers)}] "
            f"Speaker {speaker_id}"
        )
        print("=" * 70)

        # --------------------------------------------------------------
        # Already completed
        # --------------------------------------------------------------

        if speaker_id in existing_voice_ids:

            print(
                "SKIPPED: speaker already has "
                "a voice ID:"
            )

            print(
                f"  {existing_voice_ids[speaker_id]}"
            )

            skipped_count += 1

            continue

        # --------------------------------------------------------------
        # Reference directory
        # --------------------------------------------------------------

        speaker_dir = (
            reference_root
            / speaker_id
        )

        if (
            not speaker_dir.exists()
            or not speaker_dir.is_dir()
        ):

            print(
                "ERROR: no reference folder found:"
            )

            print(
                f"  {speaker_dir}"
            )

            failed_count += 1

            continue

        # --------------------------------------------------------------
        # Find reference audio
        # --------------------------------------------------------------

        clip_paths = sorted(
            f
            for f in speaker_dir.iterdir()
            if (
                f.is_file()
                and f.suffix.lower()
                in AUDIO_EXTENSIONS
            )
        )

        if not clip_paths:

            print(
                "ERROR: no audio files found in:"
            )

            print(
                f"  {speaker_dir}"
            )

            failed_count += 1

            continue

        print(
            f"Reference dir: "
            f"{speaker_dir.resolve()}"
        )

        print(
            f"Clips found  : "
            f"{len(clip_paths)}"
        )

        # --------------------------------------------------------------
        # Validate audio
        # --------------------------------------------------------------

        print()
        print(
            "Reference audio validation:"
        )

        print("-" * 70)

        valid_clip_paths = [
            path
            for path in clip_paths
            if validate_audio_file(path)
        ]

        if not valid_clip_paths:

            print(
                f"ERROR: no valid reference audio "
                f"remains for {speaker_id}."
            )

            failed_count += 1

            continue

        invalid_count = (
            len(clip_paths)
            - len(valid_clip_paths)
        )

        if invalid_count > 0:

            print()
            print(
                f"WARNING: {invalid_count} invalid "
                f"clip(s) excluded."
            )

        # --------------------------------------------------------------
        # Create voice
        # --------------------------------------------------------------

        voice_id = create_voice(
            speaker_id=speaker_id,
            clip_paths=valid_clip_paths,
            api_key=api_key,
            remove_background_noise=(
                args.remove_background_noise
            ),
        )

        if voice_id is None:

            print()
            print(
                f"FAILED: speaker {speaker_id}"
            )

            failed_count += 1

            continue

        # --------------------------------------------------------------
        # Log voice ID
        # --------------------------------------------------------------

        append_voice_id(
            voice_log_path=voice_log_path,
            speaker_id=speaker_id,
            voice_id=voice_id,
        )

        # Update in-memory state so the current run
        # also knows this speaker is complete.
        existing_voice_ids[
            speaker_id
        ] = voice_id

        created_count += 1

        print()
        print("=" * 70)
        print(
            "VOICE CREATED SUCCESSFULLY"
        )
        print("=" * 70)

        print(
            f"Speaker : {speaker_id}"
        )

        print(
            f"Voice ID: {voice_id}"
        )

        print()
        print(
            "Logged to:"
        )

        print(
            f"  {voice_log_path}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BATCH COMPLETE"
    )
    print("=" * 70)

    print(
        f"Selection           : "
        f"{batch_description}"
    )

    print(
        f"Speakers in run     : "
        f"{len(selected_speakers)}"
    )

    print(
        f"Voices created      : "
        f"{created_count}"
    )

    print(
        f"Already completed   : "
        f"{skipped_count}"
    )

    print(
        f"Failed              : "
        f"{failed_count}"
    )

    print(
        f"Voice ID log        : "
        f"{voice_log_path.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()