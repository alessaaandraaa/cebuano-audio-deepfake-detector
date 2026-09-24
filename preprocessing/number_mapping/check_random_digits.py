r"""
check_random_digits.py

Scans the ORIGINAL corpus for Random Digit utterances belonging to
speakers that have ElevenLabs voice IDs.

Only numbers 10-100 are included.

The generated TSV contains:
    speaker_id    wav_filename    number    tts_text

The original corpus is never modified.

Usage:
    python preprocessing\number_mapping\check_random_digits.py \
        --root "C:\path\to\original\bonafide" \
        --voice-ids "manifests\elevenlabs_voice_ids.txt" \
        --out "manifests\elevenlabs_number_regeneration.tsv"
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# NUMBER -> ELEVENLABS TTS MAPPING
# ---------------------------------------------------------------------------

NUMBER_MAPPING = {
    10: "dyis",
    11: "onse",
    12: "dose",
    13: "trese",
    14: "katorse",
    15: "kinse",
    16: "disisais",
    17: "disisyete",
    18: "disiotso",
    19: "disinoybe",
    20: "baynte",
    21: "baynte uno",
    22: "baynte dos",
    23: "baynte tres",
    24: "baynte kwatro",
    25: "baynte singko",
    26: "baynte sais",
    27: "baynte syete",
    28: "baynte otso",
    29: "baynte noybe",
    30: "traynta",
    31: "trayntay uno",
    32: "trayntay dos",
    33: "trayntay tres",
    34: "trayntay kwatro",
    35: "trayntay singko",
    36: "trayntay sais",
    37: "trayntay syete",
    38: "trayntay otso",
    39: "trayntay noybe",
    40: "kwarenta",
    41: "kwarentay uno",
    42: "kwarentay dos",
    43: "kwarentay tres",
    44: "kwarentay kwatro",
    45: "kwarentay singko",
    46: "kwarentay sais",
    47: "kwarentay syete",
    48: "kwarentay otso",
    49: "kwarentay noybe",
    50: "singkwenta",
    51: "singkwentay uno",
    52: "singkwentay dos",
    53: "singkwentay tres",
    54: "singkwentay kwatro",
    55: "singkwentay singko",
    56: "singkwentay sais",
    57: "singkwentay syete",
    58: "singkwentay otso",
    59: "singkwentay noybe",
    60: "saysenta",
    61: "saysentay uno",
    62: "saysentay dos",
    63: "saysentay tres",
    64: "saysentay kwatro",
    65: "saysentay singko",
    66: "saysentay sais",
    67: "saysentay syete",
    68: "saysentay otso",
    69: "saysentay noybe",
    70: "sitenta",
    71: "sitentay uno",
    72: "sitentay dos",
    73: "sitentay tres",
    74: "sitentay kwatro",
    75: "sitentay singko",
    76: "sitentay sais",
    77: "sitentay syete",
    78: "sitentay otso",
    79: "sitentay noybe",
    80: "otsenta",
    81: "otsentay uno",
    82: "otsentay dos",
    83: "otsentay tres",
    84: "otsentay kwatro",
    85: "otsentay singko",
    86: "otsentay sais",
    87: "otsentay syete",
    88: "otsentay otso",
    89: "otsentay noybe",
    90: "nobenta",
    91: "nobentay uno",
    92: "nobentay dos",
    93: "nobentay tres",
    94: "nobentay kwatro",
    95: "nobentay singko",
    96: "nobentay sais",
    97: "nobentay syete",
    98: "nobentay otso",
    99: "nobentay noybe",
    100: "usa ka gatos",
}


# ---------------------------------------------------------------------------
# VOICE IDS
# ---------------------------------------------------------------------------


def load_voice_ids(path: Path) -> dict:
    """
    Reads:

        speaker_id - voice_id

    Returns:

        {
            "0201": "voice_id_here",
            ...
        }
    """

    voice_ids = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or " - " not in line:
                continue

            speaker_id, voice_id = line.split(" - ", 1)

            voice_ids[speaker_id.strip()] = voice_id.strip()

    return voice_ids


# ---------------------------------------------------------------------------
# LOG PARSING
# ---------------------------------------------------------------------------


def find_log_file(speaker_dir: Path):
    log_files = list(speaker_dir.glob("*.log"))

    if not log_files:
        return None

    if len(log_files) > 1:
        print(
            f"WARNING: [{speaker_dir.name}] multiple .log files; "
            f"using {log_files[0].name}",
            file=sys.stderr,
        )

    return log_files[0]


def parse_random_digits(log_path: Path):
    """
    Finds Random Digit entries in a speaker's .log.

    Expected format:

        0202.111024.005622.0425.wav "Random Digit" "54"

    Returns:

        [
            ("0202.111024.005622.0425.wav", 54),
            ...
        ]
    """

    results = []

    try:
        with open(
            log_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split('"')

                if len(parts) < 5:
                    continue

                filename = parts[0].strip()
                source = parts[1].strip()
                number_text = parts[3].strip()

                if not filename.endswith(".wav"):
                    continue

                if source != "Random Digit":
                    continue

                try:
                    number = int(number_text)
                except ValueError:
                    continue

                # ONLY 10-100
                if not 10 <= number <= 100:
                    continue

                results.append((filename, number))

    except Exception as e:
        print(
            f"WARNING: failed to parse {log_path}: {e}",
            file=sys.stderr,
        )

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Find Random Digit utterances from ElevenLabs speakers "
            "and generate a regeneration TSV."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Path to ORIGINAL corpus / bonafide root containing "
            "speaker folders"
        ),
    )

    parser.add_argument(
        "--voice-ids",
        required=True,
        help="Path to ElevenLabs speaker voice ID file",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV path",
    )

    args = parser.parse_args()

    root = Path(args.root)
    voice_ids_path = Path(args.voice_ids)
    out_path = Path(args.out)

    if not root.exists():
        print(
            f"ERROR: root path does not exist: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not voice_ids_path.exists():
        print(
            f"ERROR: voice ID file does not exist: {voice_ids_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load ElevenLabs speakers
    # -----------------------------------------------------------------------

    print("Loading ElevenLabs voice IDs...")

    voice_ids = load_voice_ids(voice_ids_path)

    print(f"Loaded {len(voice_ids)} ElevenLabs speaker voice IDs.")

    # -----------------------------------------------------------------------
    # Scan only those speakers
    # -----------------------------------------------------------------------

    print(f"\nScanning original corpus:\n  {root.resolve()}")

    rows = []

    for speaker_id in sorted(voice_ids):

        speaker_dir = root / speaker_id

        if not speaker_dir.exists():
            print(
                f"WARNING: [{speaker_id}] speaker folder not found, skipping."
            )
            continue

        log_file = find_log_file(speaker_dir)

        if not log_file:
            print(f"WARNING: [{speaker_id}] no .log file found, skipping.")
            continue

        random_digits = parse_random_digits(log_file)

        if not random_digits:
            continue

        print(
            f"[{speaker_id}] found {len(random_digits)} "
            "Random Digit entries (10-100)"
        )

        for wav_filename, number in random_digits:

            tts_text = NUMBER_MAPPING.get(number)

            if tts_text is None:
                print(
                    f"WARNING: no mapping for {number}, skipping "
                    f"{speaker_id}/{wav_filename}",
                    file=sys.stderr,
                )
                continue

            rows.append(
                (
                    speaker_id,
                    wav_filename,
                    number,
                    tts_text,
                )
            )

    # -----------------------------------------------------------------------
    # Write TSV
    # -----------------------------------------------------------------------

    if not rows:
        print(
            "\nERROR: no matching Random Digit entries found.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write("speaker_id\twav_filename\tnumber\ttts_text\n")

        for speaker_id, wav_filename, number, tts_text in rows:
            f.write(f"{speaker_id}\t{wav_filename}\t{number}\t{tts_text}\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "-" * 60)

    print(f"Found {len(rows)} Random Digit files to regenerate.")

    print(f"TSV written to:\n  {out_path.resolve()}")

    print("\nPreview:")

    for row in rows[:20]:
        print(f"  {row[0]} | {row[1]} | {row[2]} -> {row[3]}")

    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")


if __name__ == "__main__":
    main()
