r"""
delete_voices.py

Deletes ONLY ElevenLabs voices whose names start with:

    ceb-

Examples:
    ceb-0201
    ceb-0202
    ceb-0399

Preset/default ElevenLabs voices are NOT touched.

Usage:
    python preprocessing\elevenlabs\delete_voices.py
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# ElevenLabs Config
# ---------------------------------------------------------------------------

VOICES_URL = "https://api.elevenlabs.io/v1/voices"
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not api_key:
        sys.exit("ERROR: ELEVENLABS_API_KEY not found.")

    headers = {
        "xi-api-key": api_key,
    }

    # -----------------------------------------------------------------------
    # Load voices
    # -----------------------------------------------------------------------

    print("Loading ElevenLabs voices...")

    try:
        response = requests.get(
            VOICES_URL,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as e:
        sys.exit(f"ERROR: Failed to connect to ElevenLabs: {e}")

    if response.status_code != 200:
        sys.exit(
            "ERROR: Failed to load voices "
            f"({response.status_code}): {response.text}"
        )

    data = response.json()

    voices = data.get("voices", [])

    # -----------------------------------------------------------------------
    # ONLY select ceb-* voices
    # -----------------------------------------------------------------------

    ceb_voices = [
        voice for voice in voices if voice.get("name", "").startswith("ceb-")
    ]

    print(f"Found {len(voices)} total voices.")

    print(f"Found {len(ceb_voices)} voices matching 'ceb-*'.")

    if not ceb_voices:
        print("Nothing to delete.")
        return

    # -----------------------------------------------------------------------
    # Show targets
    # -----------------------------------------------------------------------

    print("\nVOICES THAT WILL BE DELETED:")
    print("-" * 60)

    for voice in ceb_voices:
        print(f"{voice.get('name')} | {voice.get('voice_id')}")

    print("-" * 60)

    # -----------------------------------------------------------------------
    # Confirm
    # -----------------------------------------------------------------------

    confirm = input(
        f"\nDELETE THESE {len(ceb_voices)} ceb-* VOICES? "
        "Type 'DELETE' to continue: "
    )

    if confirm != "DELETE":
        print("Cancelled.")
        return

    # -----------------------------------------------------------------------
    # Delete
    # -----------------------------------------------------------------------

    deleted = 0
    failed = 0

    for i, voice in enumerate(
        ceb_voices,
        start=1,
    ):

        voice_id = voice.get("voice_id")
        voice_name = voice.get("name", "<unnamed>")

        if not voice_id:
            print(
                f"[{i}/{len(ceb_voices)}] "
                f"SKIPPED: {voice_name} has no voice ID"
            )

            failed += 1
            continue

        print(f"[{i}/{len(ceb_voices)}] Deleting {voice_name}...")

        try:
            response = requests.delete(
                f"{VOICES_URL}/{voice_id}",
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as e:
            print(f"  FAILED: {e}")

            failed += 1
            continue

        if response.status_code in (200, 204):

            print("  Deleted.")
            deleted += 1

        else:

            print(f"  FAILED ({response.status_code}): {response.text}")

            failed += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "-" * 60)

    print(f"Done. {deleted} ceb-* voices deleted, {failed} failed.")

    print(f"Preset/default voices were NOT targeted.")


if __name__ == "__main__":
    main()
