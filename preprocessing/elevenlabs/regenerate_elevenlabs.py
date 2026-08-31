
"""
regenerate_elevenlabs.py

Regenerates ONLY the utterances listed in a TSV file.

TSV format:
    speaker_id    wav_filename    transcript    saying

Example:
    0203    0203.111024.084752.0378.wav    1    uno
    0206    0206.111024.094638.0373.wav    1    uno
    0213    0213.111025.023930.0471.wav    2    dos

The TSV is the source of truth for:
    - speaker_id  -> ElevenLabs voice
    - wav_filename -> output filename
    - saying -> text sent to ElevenLabs

If a speaker has no ElevenLabs voice ID, that row is skipped.

Output:
    data/processed/elevenlabs/<speaker>/<wav_stem>.2.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "elevenlabs"
)

VOICE_ID_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "elevenlabs_voice_ids.txt"
)


# ============================================================
# ELEVENLABS CONFIG
# ============================================================

API_URL_TEMPLATE = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
)

MODEL_ID = "eleven_v3"
LANGUAGE_CODE = "ceb"

REQUEST_TIMEOUT = 120
REQUEST_DELAY = 0.25


# ============================================================
# ERRORS
# ============================================================

class QuotaExceededError(Exception):
    pass


# ============================================================
# ELEVENLABS HELPERS
# ============================================================

def is_quota_exceeded(response: requests.Response) -> bool:
    """
    Detect ElevenLabs quota/credit exhaustion.
    """

    if response.status_code not in (401, 429):
        return False

    body = response.text.lower()

    return (
        "quota_exceeded" in body
        or (
            "insufficient" in body
            and "credit" in body
        )
    )


def load_voice_ids() -> dict:
    """
    Loads:

        speaker_id - elevenlabs_voice_id

    from:

        manifests/elevenlabs_voice_ids.txt
    """

    if not VOICE_ID_FILE.exists():
        sys.exit(
            f"ERROR: Voice ID file not found:\n"
            f"  {VOICE_ID_FILE}"
        )

    ids = {}

    with open(VOICE_ID_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if " - " not in line:
                continue

            speaker_id, voice_id = line.split(
                " - ",
                1
            )

            speaker_id = speaker_id.strip()
            voice_id = voice_id.strip()

            if speaker_id and voice_id:
                ids[speaker_id] = voice_id

    return ids


def generate_audio(
    api_key: str,
    voice_id: str,
    text: str,
):
    """
    Sends text to ElevenLabs and returns WAV bytes.
    """

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
            API_URL_TEMPLATE.format(
                voice_id=voice_id
            ),
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as e:

        print(f"    REQUEST ERROR: {e}")
        return None

    if is_quota_exceeded(response):
        raise QuotaExceededError(
            response.text
        )

    if response.status_code != 200:

        print(
            f"    ElevenLabs error "
            f"{response.status_code}: "
            f"{response.text[:300]}"
        )

        return None

    return response.content


# ============================================================
# TSV LOADER
# ============================================================

def load_targets(tsv_path: Path) -> list:
    """
    Loads rows from:

        speaker_id    wav_filename    transcript    saying

    Handles both:
        - actual TSV files
        - tabs with extra whitespace around fields
        - a header row

    Returns:

        [
            {
                "speaker_id": "...",
                "wav_filename": "...",
                "transcript": "...",
                "saying": "..."
            }
        ]
    """

    targets = []

    with open(
        tsv_path,
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_number, raw_line in enumerate(
            f,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Split on TAB
            # ------------------------------------------------

            parts = line.split("\t")

            # ------------------------------------------------
            # Skip header
            # ------------------------------------------------

            if (
                parts
                and parts[0].strip().lower()
                == "speaker_id"
            ):
                continue

            if len(parts) < 4:

                print(
                    f"WARNING: malformed TSV line "
                    f"{line_number}, skipping:"
                )

                print(
                    repr(raw_line.rstrip("\n"))
                )

                continue

            speaker_id = parts[0].strip()
            wav_filename = parts[1].strip()
            transcript = parts[2].strip()
            saying = parts[3].strip()

            # ------------------------------------------------
            # Basic validation
            # ------------------------------------------------

            if not speaker_id:
                print(
                    f"WARNING: line {line_number} "
                    f"has no speaker ID, skipping."
                )
                continue

            if not wav_filename:
                print(
                    f"WARNING: line {line_number} "
                    f"has no WAV filename, skipping."
                )
                continue

            if not transcript:
                print(
                    f"WARNING: line {line_number} "
                    f"has no transcript, skipping."
                )
                continue

            if not saying:
                print(
                    f"WARNING: line {line_number} "
                    f"has no saying, skipping."
                )
                continue

            targets.append(
                {
                    "speaker_id": speaker_id,
                    "wav_filename": wav_filename,
                    "transcript": transcript,
                    "saying": saying,
                }
            )

    return targets


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate selected ElevenLabs "
            "utterances from a TSV."
        )
    )

    parser.add_argument(
        "tsv",
        type=Path,
        help=(
            "TSV containing "
            "speaker_id, wav_filename, "
            "transcript, saying"
        ),
    )

    args = parser.parse_args()

    tsv_path = args.tsv

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.environ.get(
        "ELEVENLABS_API_KEY"
    )

    if not api_key:

        sys.exit(
            "ERROR: ELEVENLABS_API_KEY "
            "not found in .env"
        )

    # --------------------------------------------------------
    # TSV existence
    # --------------------------------------------------------

    if not tsv_path.exists():

        sys.exit(
            f"ERROR: TSV not found:\n"
            f"  {tsv_path}"
        )

    # --------------------------------------------------------
    # Load voices
    # --------------------------------------------------------

    print("Loading ElevenLabs voice IDs...")

    voice_ids = load_voice_ids()

    print(
        f"Loaded {len(voice_ids)} speaker voice IDs."
    )

    # --------------------------------------------------------
    # Load targets
    # --------------------------------------------------------

    print(
        f"\nLoading regeneration list:\n"
        f"  {tsv_path}"
    )

    targets = load_targets(tsv_path)

    if not targets:

        sys.exit(
            "ERROR: No valid rows found in TSV."
        )

    print(
        f"Found {len(targets)} "
        f"utterances to regenerate.\n"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    generated = 0
    skipped_no_voice = 0
    failed = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for i, target in enumerate(
        targets,
        start=1,
    ):

        speaker_id = target["speaker_id"]
        wav_filename = target["wav_filename"]
        transcript = target["transcript"]
        saying = target["saying"]

        print(
            "--------------------------------------------------"
        )

        print(
            f"[{i}/{len(targets)}] "
            f"SPEAKER: {speaker_id}"
        )

        print(
            f"    WAV:        {wav_filename}"
        )

        print(
            f"    NUMBER:     {transcript}"
        )

        print(
            f"    SAYING:     {saying}"
        )

        # ----------------------------------------------------
        # Check voice
        # ----------------------------------------------------

        if speaker_id not in voice_ids:

            print(
                f"    SKIPPED: no ElevenLabs voice ID "
                f"for speaker {speaker_id}"
            )

            skipped_no_voice += 1
            continue

        voice_id = voice_ids[speaker_id]

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        wav_stem = Path(
            wav_filename
        ).stem

        output_path = (
            OUTPUT_ROOT
            / speaker_id
            / f"{wav_stem}.2.wav"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            audio = generate_audio(
                api_key=api_key,
                voice_id=voice_id,
                text=saying,
            )

            if audio is None:

                print(
                    "    FAILED: ElevenLabs "
                    "returned no audio."
                )

                failed += 1
                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                output_path,
                "wb",
            ) as f:

                f.write(audio)

            print(
                f"    SAVED: {output_path.name}"
            )

            generated += 1

            # ------------------------------------------------
            # Avoid hammering API
            # ------------------------------------------------

            time.sleep(
                REQUEST_DELAY
            )

        except QuotaExceededError:

            print(
                "\n"
                "=================================================="
            )

            print(
                "!!! STOPPED: ElevenLabs credits ran out !!!"
            )

            print(
                "=================================================="
            )

            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        "=================================================="
    )

    print("DONE")

    print(
        f"Generated:         {generated}"
    )

    print(
        f"No voice ID:       {skipped_no_voice}"
    )

    print(
        f"Failed:            {failed}"
    )

    print(
        f"Total TSV rows:    {len(targets)}"
    )

    print(
        "=================================================="
    )


if __name__ == "__main__":
    main()
