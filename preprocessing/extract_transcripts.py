"""
extract_transcripts.py

Walks EVERY speaker folder under --root, parses each speaker's .log file,
cleans each utterance's transcript text for TTS input, and writes ONE .txt
file per speaker.

Output structure:

    root/0204/0204.log
    -> out/0204.txt

Each line in the output represents one utterance.

Cleaning rules:
  - Strips parenthetical English glosses/translations
  - Collapses stray tabs / repeated whitespace
  - Handles "Random Digit" entries according to DIGIT_MODE

Excluded utterances are logged with:
  - Speaker ID
  - WAV filename
  - Source file
  - Reason for exclusion

Usage:
    python extract_transcripts.py --root "C:\\path\\to\\bonafide" --out transcripts
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# TOGGLES
# ---------------------------------------------------------------------------

# Source-file prefixes to exclude entirely.
EXCLUDE_SOURCE_PREFIXES = (
    # "TGL_",
)

# How to handle "Random Digit" entries.
#
# "keep"    -> keep the digit text as-is
# "exclude" -> skip Random Digit lines
DIGIT_MODE = "keep"

# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(r'^(\S+\.wav)\s+"([^"]*)"\s+(.*)$')
PAREN_PATTERN = re.compile(r'\s*\([^)]*\)')


def clean_text(raw: str) -> str:
    text = raw.strip()

    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]

    text = PAREN_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"-{2,}", " ", text)

    return text


def find_log_file(speaker_dir: Path):
    log_files = list(speaker_dir.glob("*.log"))

    if not log_files:
        return None

    if len(log_files) > 1:
        print(
            f"WARNING: multiple .log files in {speaker_dir}, "
            f"using {log_files[0].name}",
            file=sys.stderr,
        )

    return log_files[0]


def parse_transcript_lines(log_path: Path):
    """
    Yields:
        (source_wav_filename, source_file, raw_text)
    for each transcript line.
    """

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            # Metadata line, not a transcript line.
            if "=" in line and not line.strip().startswith('"'):
                continue

            m = LINE_PATTERN.match(line.strip())

            if not m:
                continue

            filename, source_file, rest = m.groups()

            yield filename, source_file, rest


def should_exclude_source(source_file: str) -> bool:
    return any(
        source_file.startswith(prefix)
        for prefix in EXCLUDE_SOURCE_PREFIXES
    )


def is_digit_entry(source_file: str) -> bool:
    return source_file.strip().lower() == "random digit"


def log_exclusion(
    speaker_id: str,
    filename: str,
    source_file: str,
    reason: str,
):
    print(
        f"EXCLUDED [{speaker_id}] "
        f"{filename} | source={source_file!r} | reason={reason}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract + clean utterances into one .txt file per speaker"
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Path to bonafide root folder",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output folder for speaker transcript files",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    if not root.exists():
        print(
            f"ERROR: root not found: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {out_root.resolve()}")

    speaker_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir()
    )

    if not speaker_dirs:
        print(
            f"ERROR: no speaker subfolders found under {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(speaker_dirs)} speaker folders under {root}")
    print(f"Writing transcripts to: {out_root.resolve()}")
    print(f"EXCLUDE_SOURCE_PREFIXES = {EXCLUDE_SOURCE_PREFIXES}")
    print(f"DIGIT_MODE = {DIGIT_MODE!r}")
    print("-" * 60)

    total_written = 0
    total_excluded_source = 0
    total_excluded_digit = 0
    total_excluded_empty = 0
    total_malformed = 0

    missing_logs = []

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        log_file = find_log_file(speaker_dir)

        if not log_file:
            missing_logs.append(speaker_id)

            print(
                f"WARNING: [{speaker_id}] no .log file found",
                file=sys.stderr,
            )

            continue

        # One output file per speaker.
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"{speaker_id}.txt"

        speaker_utterances = []

        for filename, source_file, raw_text in parse_transcript_lines(log_file):

            # ---------------------------------------------------------------
            # Source exclusion
            # ---------------------------------------------------------------

            if should_exclude_source(source_file):
                total_excluded_source += 1

                log_exclusion(
                    speaker_id,
                    filename,
                    source_file,
                    "source prefix excluded",
                )

                continue

            # ---------------------------------------------------------------
            # Digit exclusion
            # ---------------------------------------------------------------

            if is_digit_entry(source_file) and DIGIT_MODE == "exclude":
                total_excluded_digit += 1

                log_exclusion(
                    speaker_id,
                    filename,
                    source_file,
                    "Random Digit excluded",
                )

                continue

            # ---------------------------------------------------------------
            # Clean transcript
            # ---------------------------------------------------------------

            cleaned = clean_text(raw_text)

            if not cleaned:
                total_excluded_empty += 1

                log_exclusion(
                    speaker_id,
                    filename,
                    source_file,
                    "empty after cleaning",
                )

                continue

            # Keep one utterance per line.
            speaker_utterances.append(cleaned)
            total_written += 1

        # ---------------------------------------------------------------
        # Write one file for the entire speaker
        # ---------------------------------------------------------------

        if speaker_utterances:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(speaker_utterances))
                f.write("\n")

            print(
                f"[{speaker_id}] wrote "
                f"{len(speaker_utterances)} utterances -> {out_path}"
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    if missing_logs:
        print(
            f"WARNING: {len(missing_logs)} speaker(s) had no .log file: "
            f"{missing_logs[:10]}"
            f"{'...' if len(missing_logs) > 10 else ''}",
            file=sys.stderr,
        )

    print("-" * 60)
    print("Done.")
    print(f"Utterances written          : {total_written}")
    print(f"Excluded (source prefix)   : {total_excluded_source}")
    print(f"Excluded (digit mode)      : {total_excluded_digit}")
    print(f"Excluded (empty after clean): {total_excluded_empty}")
    print(f"Speakers processed         : {len(speaker_dirs) - len(missing_logs)}")
    print(f"Speakers missing .log      : {len(missing_logs)}")
    print(f"\nOutput folder: {out_root.resolve()}")


if __name__ == "__main__":
    main()