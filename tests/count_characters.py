r"""
count_characters.py

Scans an entire dataset folder recursively for .log files and counts
the characters in utterances contained in those logs.

Expected log format:

    root/
      speaker_0268/
        speaker_0268.log
        ...
      speaker_0269/
        speaker_0269.log
        ...

Utterance lines are expected to look like:

    0268.111028.015536.0363.wav "CEB_Utt_CommonExpressions.txt" "Únsa ang átong pagkáon karón?"

Metadata lines such as:

    SpeakerID = "0268"
    SpeakerGender = "female"

are ignored.

--root is OPTIONAL. It defaults to PLD_CORPUS_ROOT, which this script
reads from the .env file at the project root (copy .env.example).

Note: .env is read by this script, NOT by your shell. Do NOT write
"$env:PLD_CORPUS_ROOT" on the command line -- PowerShell expands that to an
empty string and argparse fails with "expected one argument". Just omit
--root and let the script read .env.

Usage (PowerShell):

    python tests\count_characters.py

    # point at a different dataset root for one run
    python tests\count_characters.py --root "D:\other\CEB"

"""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from collections import Counter

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}


# Matches an audio filename followed by:
#   "source file"
#   "utterance"
#
# Example:
# 0268.111028.015536.0363.wav "CEB_Utt_CommonExpressions.txt" "Únsa ang átong pagkáon karón?"
UTTERANCE_PATTERN = re.compile(
    r'^(.+?\.(?:wav|flac|mp3|ogg|m4a))\s+"[^"]+"\s+"(.*)"\s*$',
    re.IGNORECASE,
)


def find_log_files(root: Path):
    """Recursively finds all .log files under root."""
    return sorted(
        f
        for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() == ".log"
    )


def parse_log_file(log_path: Path):
    """
    Parses a single .log file.

    Returns:
        utterance_count
        character_count
    """

    utterance_count = 0
    character_count = 0

    try:
        with log_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:

            for line in f:
                line = line.rstrip("\r\n")

                match = UTTERANCE_PATTERN.match(line)

                if not match:
                    continue

                utterance = match.group(2)

                utterance_count += 1
                character_count += len(utterance)

    except Exception as e:
        print(
            f"WARNING: failed to read {log_path}: {e}",
            file=sys.stderr,
        )

    return utterance_count, character_count


def scan_dataset(root: Path):
    """
    Walks the entire dataset recursively, finds every .log file,
    and counts characters in its utterances.
    """

    log_files = find_log_files(root)

    if not log_files:
        print(
            f"WARNING: no .log files found under {root}.",
            file=sys.stderr,
        )

    total_utterances = 0
    total_characters = 0

    per_log = []

    for log_file in log_files:
        utterances, characters = parse_log_file(log_file)

        total_utterances += utterances
        total_characters += characters

        per_log.append(
            {
                "path": log_file,
                "utterances": utterances,
                "characters": characters,
            }
        )

    return (
        log_files,
        per_log,
        total_utterances,
        total_characters,
    )


def print_summary(
    root: Path,
    log_files,
    per_log,
    total_utterances,
    total_characters,
):
    """Prints a summary of the character inventory."""

    print("=" * 70)
    print(f"Character inventory: {root}")
    print("=" * 70)

    print(f"Log files found      : {len(log_files):,}")
    print(f"Total utterances     : {total_utterances:,}")
    print(f"Total characters     : {total_characters:,}")

    if total_utterances:
        average = total_characters / total_utterances
        print(f"Average chars/utter. : {average:.2f}")

    print("=" * 70)

    if per_log:
        print("\nPer-log breakdown:")
        print("-" * 70)

        for item in per_log:
            print(
                f"{item['path']}\n"
                f"  Utterances : {item['utterances']:,}\n"
                f"  Characters : {item['characters']:,}"
            )

    # Show logs with the most characters.
    if per_log:
        print("\nLogs with most characters:")
        print("-" * 70)

        largest = sorted(
            per_log,
            key=lambda x: x["characters"],
            reverse=True,
        )

        for item in largest[:10]:
            print(
                f"  {item['characters']:>10,} chars | "
                f"{item['utterances']:>6,} utterances | "
                f"{item['path']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Count characters in utterances across an entire dataset"
    )

    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Path to the dataset root folder. "
            "Default: PLD_CORPUS_ROOT from .env"
        ),
    )

    args = parser.parse_args()

    root_raw = args.root or os.environ.get("PLD_CORPUS_ROOT")
    if not root_raw:
        print(
            "ERROR: no corpus root.\n"
            "  Either pass --root, or copy .env.example to .env and set\n"
            "  PLD_CORPUS_ROOT to the ORIGINAL PLD CEB corpus (the folder\n"
            "  holding one directory per speaker).",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(root_raw)

    if not root.exists():
        print(
            f"ERROR: root path does not exist: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not root.is_dir():
        print(
            f"ERROR: root path is not a directory: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    (
        log_files,
        per_log,
        total_utterances,
        total_characters,
    ) = scan_dataset(root)

    print_summary(
        root,
        log_files,
        per_log,
        total_utterances,
        total_characters,
    )


if __name__ == "__main__":
    main()
