"""
determine_number_logs.py

Scans ALL speakers in the original OpenSLR CEB corpus.

For every speaker directory, finds all "Random Digit" entries,
detects the number in the transcript, translates it using
numbers.json, and creates a targeted manifest.

Output:

    manifests/transcripts_regenerate.txt

Format:

    SPEAKER_ID<TAB>NUMBER<TAB>WAV_FILENAME<TAB>CEBUANO_TEXT

Example:

    0201    1    0201.abc.wav    usa
    0237    1    0237.xyz.wav    usa
    0201    2    0201.def.wav    duha
    ...
    0205    100  0205.foo.wav    usa ka gatos

IMPORTANT:
    - Scans ALL speakers in the corpus.
    - Does NOT use the ElevenLabs speaker manifest.
    - Preserves the exact original WAV filename.
    - Sorts the final output numerically by number: 1 -> 100.
    - If a number occurs multiple times, every occurrence is kept.
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = Path("C:/Users/Ninzz/Programming/PLD/up-dsp-pld/PLD/CEB")

NUMBERS_JSON_PATH = PROJECT_ROOT / "preprocessing" / "numbers.json"

OUT_FILE = PROJECT_ROOT / "manifests" / "transcripts_regenerate.txt"


# ---------------------------------------------------------------------------
# RAW LOG FORMAT
# ---------------------------------------------------------------------------

# Expected:
#
# filename.wav "source" Transcript
#
# Example:
#
# 0201.abc.wav "Random Digit" 12
#
LINE_PATTERN = re.compile(r'^(\S+\.wav)\s+"([^"]*)"\s+(.*)$')


# ---------------------------------------------------------------------------
# LOG FILE
# ---------------------------------------------------------------------------


def find_log_file(speaker_dir: Path) -> Path | None:
    """
    Find the .log file directly inside a speaker folder.

    If multiple .log files exist, the first one returned by
    the filesystem is used and a warning is printed.
    """

    log_files = list(speaker_dir.glob("*.log"))

    if not log_files:
        return None

    if len(log_files) > 1:
        print(
            "WARNING: multiple .log files in "
            f"{speaker_dir}, using "
            f"{log_files[0].name}",
            file=sys.stderr,
        )

    return log_files[0]


# ---------------------------------------------------------------------------
# SPEAKER DISCOVERY
# ---------------------------------------------------------------------------


def find_all_speakers() -> list[Path]:
    """
    Find ALL speaker directories directly inside RAW_ROOT.

    Example:

        CEB/
            0201/
            0202/
            0203/
            ...

    No speaker manifest is used.
    """

    if not RAW_ROOT.exists():

        print(
            f"ERROR: corpus root not found:\n  {RAW_ROOT}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not RAW_ROOT.is_dir():

        print(
            f"ERROR: corpus root is not a directory:\n  {RAW_ROOT}",
            file=sys.stderr,
        )

        sys.exit(1)

    speaker_dirs = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())

    if not speaker_dirs:

        print(
            f"ERROR: no speaker directories found in:\n  {RAW_ROOT}",
            file=sys.stderr,
        )

        sys.exit(1)

    return speaker_dirs


# ---------------------------------------------------------------------------
# NUMBER MAPPING
# ---------------------------------------------------------------------------


def load_number_mapping() -> dict[str, str]:
    """
    Load numbers.json.

    Expected:

        {
            "1": "usa",
            "2": "duha",
            ...
            "100": "usa ka gatos"
        }
    """

    if not NUMBERS_JSON_PATH.exists():

        print(
            f"ERROR: {NUMBERS_JSON_PATH} not found.",
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        with open(
            NUMBERS_JSON_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            mapping = json.load(f)

    except Exception as e:

        print(
            f"ERROR: failed to load {NUMBERS_JSON_PATH}: {e}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not isinstance(mapping, dict):

        print(
            "ERROR: numbers.json must contain a JSON object.",
            file=sys.stderr,
        )

        sys.exit(1)

    return mapping


# ---------------------------------------------------------------------------
# NUMBER DETECTION
# ---------------------------------------------------------------------------


def detect_numbers(
    text: str,
    number_mapping: dict[str, str],
) -> list[str]:
    """
    Detect numeric values appearing in the Random Digit transcript.

    Returns the numbers in the order they appear.

    Example:

        "12"

    ->

        ["12"]
    """

    matches = re.findall(
        r"\b\d+\b",
        text,
    )

    detected = []

    for number in matches:

        if number in number_mapping:

            detected.append(number)

        else:

            print(
                f"WARNING: number {number!r} not found in numbers.json",
                file=sys.stderr,
            )

    return detected


# ---------------------------------------------------------------------------
# NUMBER EXPANSION
# ---------------------------------------------------------------------------


def replace_numbers(
    text: str,
    number_mapping: dict[str, str],
) -> str:
    """
    Replace numeric values with their Cebuano
    representations.

    Longer numbers are replaced first.

    Example:

        "21"

    ->

        "baynte-usa"
    """

    compiled_numbers = [
        (
            re.compile(rf"\b{re.escape(number)}\b"),
            word,
        )
        for number, word in sorted(
            number_mapping.items(),
            key=lambda x: int(x[0]),
            reverse=True,
        )
    ]

    for pattern, word in compiled_numbers:

        text = pattern.sub(
            word,
            text,
        )

    return text.strip()


# ---------------------------------------------------------------------------
# PROCESS ONE SPEAKER
# ---------------------------------------------------------------------------


def process_speaker(
    speaker_dir: Path,
    number_mapping: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """
    Process one speaker directory.

    Returns:

        [
            (
                speaker_id,
                number,
                wav_filename,
                expanded_text,
            ),
            ...
        ]
    """

    speaker_id = speaker_dir.name

    log_file = find_log_file(speaker_dir)

    if not log_file:

        print(f"[{speaker_id}] NO LOG FILE")

        return []

    results = []

    with open(
        log_file,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            match = LINE_PATTERN.match(line)

            if not match:
                continue

            wav_name, source, text = match.groups()

            # ---------------------------------------------------------------
            # Only Random Digit entries
            # ---------------------------------------------------------------

            if source.strip().lower() != "random digit":
                continue

            # ---------------------------------------------------------------
            # Remove surrounding quotes if present
            # ---------------------------------------------------------------

            if text.startswith('"') and text.endswith('"') and len(text) >= 2:

                text = text[1:-1]

            text = text.strip()

            # ---------------------------------------------------------------
            # Detect number BEFORE replacing it.
            # ---------------------------------------------------------------

            detected_numbers = detect_numbers(
                text,
                number_mapping,
            )

            if not detected_numbers:

                print(
                    "WARNING: "
                    f"{speaker_id}/"
                    f"{log_file.name}:"
                    f"{line_number}: "
                    "Random Digit entry contains "
                    "no recognized number: "
                    f"{text!r}",
                    file=sys.stderr,
                )

                continue

            # ---------------------------------------------------------------
            # Expand number into Cebuano
            # ---------------------------------------------------------------

            expanded_text = replace_numbers(
                text,
                number_mapping,
            )

            # ---------------------------------------------------------------
            # Preserve EVERY occurrence.
            # ---------------------------------------------------------------

            for number in detected_numbers:

                results.append(
                    (
                        speaker_id,
                        number,
                        wav_name,
                        expanded_text,
                    )
                )

    print(f"[{speaker_id}] Random Digit entries: {len(results)}")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():

    print("=" * 70)
    print("DETECTING RANDOM DIGIT UTTERANCES")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Find ALL speakers
    # -----------------------------------------------------------------------

    speaker_dirs = find_all_speakers()

    print(f"Corpus root         : {RAW_ROOT}")

    print(f"Speaker directories : {len(speaker_dirs)}")

    # -----------------------------------------------------------------------
    # Load number mapping
    # -----------------------------------------------------------------------

    number_mapping = load_number_mapping()

    print(f"Number mappings     : {len(number_mapping)}")

    print("-" * 70)

    # -----------------------------------------------------------------------
    # Process every speaker
    # -----------------------------------------------------------------------

    to_regenerate = []

    speakers_processed = 0
    speakers_without_logs = 0

    for speaker_dir in speaker_dirs:

        results = process_speaker(
            speaker_dir=speaker_dir,
            number_mapping=number_mapping,
        )

        if results:
            speakers_processed += 1

        else:
            log_file = find_log_file(speaker_dir)

            if log_file is None:
                speakers_without_logs += 1

        to_regenerate.extend(results)

    # -----------------------------------------------------------------------
    # SORT NUMERICALLY: 1 -> 100
    # -----------------------------------------------------------------------

    to_regenerate.sort(
        key=lambda item: (
            int(item[1]),  # number
            item[0],  # speaker ID
            item[2],  # WAV filename
        )
    )

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------

    OUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for (
            speaker_id,
            number,
            wav_name,
            expanded_text,
        ) in to_regenerate:

            f.write(f"{speaker_id}\t{number}\t{wav_name}\t{expanded_text}\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Speaker directories   : {len(speaker_dirs)}")

    print(f"Speakers with results : {speakers_processed}")

    print(f"Speakers without logs : {speakers_without_logs}")

    print(f"Random Digit entries  : {len(to_regenerate)}")

    print(f"Output                : {OUT_FILE}")

    print()
    print("Output is sorted numerically from number 1 through 100.")

    print("=" * 70)


if __name__ == "__main__":
    main()
