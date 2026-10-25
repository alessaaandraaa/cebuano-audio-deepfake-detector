r"""
find_nonrandom_numbers.py

Scans speaker .log files for utterances that contain a digit/number
OUTSIDE of "Random Digit" source entries — i.e. numbers embedded in
ordinary sentences, isolated prompts, etc. rather than the dedicated
digit-reading recordings.

Limited to only the speakers listed in elevenlabs_selected_speakers.csv
(or any CSV with a speaker_id column).

Why this matters: extract_transcripts.py's single-digit (1-9) manifest
lookup ONLY applies to Random Digit entries; a bare digit or a 10-100
number embedded in a non-Random-Digit sentence still gets expanded via
the fixed numbers.json mapping (or left as a literal digit character if
under 10 and not Random Digit — see extract_transcripts.py). This
script tells you how many such utterances actually exist and what they
look like, so you can decide whether that matters for your corpus.

Usage:
    python preprocessing/find_nonrandom_numbers.py \
        --root "C:\Users\Ninzz\Programming\PLD\up-dsp-pld\PLD\CEB" \
        --speakers-csv manifests\elevenlabs_selected_speakers.csv \
        --out nonrandom_numbers.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Same parsing rules as extract_transcripts.py
# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(r'^(\S+\.wav)\s+"([^"]*)"\s+(.*)$')

# Matches any run of digits, anywhere in the text (not just standalone
# 1-9) — this is deliberately broader than extract_transcripts.py's
# SINGLE_DIGIT_PATTERN, since we want to catch 10-100 numbers too.
ANY_NUMBER_PATTERN = re.compile(r"\d+")

EXCLUDE_SOURCE_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)


def is_digit_entry(source_file: str) -> bool:
    return source_file.strip().lower() == "random digit"


def should_exclude_source(source_file: str) -> bool:
    return any(
        source_file.startswith(prefix) for prefix in EXCLUDE_SOURCE_PREFIXES
    )


def load_speaker_ids(csv_path: Path) -> list[str]:
    speakers = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            speaker_id = row[0].strip()
            if not speaker_id:
                continue
            if line_number == 1 and speaker_id.lower() in {
                "speaker_id",
                "speaker",
                "id",
            }:
                continue
            speakers.append(speaker_id)
    return list(dict.fromkeys(speakers))


def find_log_file(speaker_dir: Path):
    log_files = list(speaker_dir.glob("*.log"))
    if not log_files:
        return None
    return log_files[0]


def parse_transcript_lines(log_path: Path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "=" in line and not line.strip().startswith('"'):
                continue
            match = LINE_PATTERN.match(line.strip())
            if not match:
                continue
            yield match.groups()  # filename, source_file, raw_text


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find non-Random-Digit utterances containing numbers, "
            "for selected speakers only"
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Path to the corpus root folder "
            "(speaker subfolders with .log files)"
        ),
    )
    parser.add_argument(
        "--speakers-csv",
        required=True,
        help=(
            "CSV with a speaker_id column "
            "(e.g. elevenlabs_selected_speakers.csv)"
        ),
    )
    parser.add_argument(
        "--out", default=None, help="Optional path to write results as CSV"
    )
    parser.add_argument(
        "--include-excluded-sources",
        action="store_true",
        help=(
            "Also include utterances from TGL_/CEB_Utt_Eng-prefixed "
            "sources (excluded from transcripts by default)"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    speakers_csv = Path(args.speakers_csv)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)
    if not speakers_csv.exists():
        print(
            f"ERROR: speakers CSV not found: {speakers_csv}", file=sys.stderr
        )
        sys.exit(1)

    speaker_ids = load_speaker_ids(speakers_csv)
    print(f"Speakers to check: {len(speaker_ids)}")
    print("-" * 70)

    results = []
    missing_logs = []
    missing_dirs = []

    for speaker_id in speaker_ids:
        speaker_dir = root / speaker_id

        if not speaker_dir.exists():
            missing_dirs.append(speaker_id)
            continue

        log_file = find_log_file(speaker_dir)
        if not log_file:
            missing_logs.append(speaker_id)
            continue

        for filename, source_file, raw_text in parse_transcript_lines(
            log_file
        ):

            if is_digit_entry(source_file):
                # this is exactly what we're excluding -- that's the point
                continue

            if (
                should_exclude_source(source_file)
                and not args.include_excluded_sources
            ):
                # wouldn't end up in transcripts anyway, skip by default
                continue

            numbers_found = ANY_NUMBER_PATTERN.findall(raw_text)
            if not numbers_found:
                continue

            results.append(
                {
                    "speaker_id": speaker_id,
                    "wav_filename": filename,
                    "source_file": source_file,
                    "numbers_found": ",".join(numbers_found),
                    "text": raw_text,
                }
            )

    print(
        f"Found {len(results)} non-Random-Digit utterance(s) containing a"
        " number."
    )
    print("-" * 70)

    for r in results:
        print(f"[{r['speaker_id']}] {r['wav_filename']}")
        print(f"  source : {r['source_file']}")
        print(f"  numbers: {r['numbers_found']}")
        print(f"  text   : {r['text']}")
        print()

    if missing_dirs:
        print(
            f"WARNING: {len(missing_dirs)} speaker(s) had no folder under"
            f" root: {missing_dirs}",
            file=sys.stderr,
        )
    if missing_logs:
        print(
            f"WARNING: {len(missing_logs)} speaker(s) had no .log file:"
            f" {missing_logs}",
            file=sys.stderr,
        )

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "speaker_id",
                    "wav_filename",
                    "source_file",
                    "numbers_found",
                    "text",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {out_path.resolve()}")

    print()
    print("=" * 70)
    print(f"Total non-Random-Digit utterances with a number: {len(results)}")
    print(
        "Speakers checked:"
        f" {len(speaker_ids) - len(missing_dirs) - len(missing_logs)} /"
        f" {len(speaker_ids)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
