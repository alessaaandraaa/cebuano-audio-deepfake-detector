"""
extract_transcripts.py

Extracts and performs corpus-level cleanup on utterance transcripts
from each speaker's .log file.

The script writes ONE .txt file per speaker.

Input:
    root/
      0201/
        0201.log
        *.wav
      0202/
        0202.log
        *.wav

Output:
    transcripts/
      0201.txt
      0202.txt

IMPORTANT:
    This script performs corpus-level cleanup to establish the
    GROUND TRUTH vocabulary distribution for the deepfake dataset.

    It intentionally does NOT perform MMS-specific vocoder normalization
    (like stripping diacritics or comma-padding). That belongs in
    generate_mms.py.

    It DOES:
  - Remove parenthetical English glosses/translations.
  - Exclude unwanted transcript sources.
  - Handle Random Digit entries according to DIGIT_MODE.
  - Expand raw digits using two separate sources, since single digits
    (1-9) do NOT have a fixed pronunciation across speakers (e.g. "uno"
    vs "usa", "dos" vs "duha", "sais" vs "unom") while 10-100 do:
      * 1-9   -> ONLY expanded when the transcript's source is a
                 "Random Digit" entry (per is_digit_entry()). Looked up
                 PER UTTERANCE (by wav_filename) from
                 manifests/original_pronunciations.tsv, so each
                 recording keeps the pronunciation that was actually
                 spoken in it. A bare 1-9 digit appearing in a
                 non-Random-Digit utterance is left untouched, since
                 the per-utterance manifest was only built to cover
                 Random Digit recordings.
      * 10-100 -> expanded via preprocessing/numbers.json, a fixed
                 corpus-wide mapping, applied regardless of source.
  - Collapse stray/repeated whitespace.
  - Preserve capitalization and original wording.

Usage:

    python preprocessing/extract_transcripts.py \
        --root "C:/Users/Ninzz/Programming/PLD/up-dsp-pld/PLD/CEB" \
        --out transcripts
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# TOGGLES
# ---------------------------------------------------------------------------

EXCLUDE_SOURCE_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)

DIGIT_MODE = "keep"


# ---------------------------------------------------------------------------
# REGEX & NUMBER EXPANSION
# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(r'^(\S+\.wav)\s+"([^"]*)"\s+(.*)$')

PAREN_PATTERN = re.compile(r"\s*\([^)]*\)")

# Matches a standalone single digit 1-9 (not part of a longer digit run,
# e.g. it will NOT match the "1" in "10").
SINGLE_DIGIT_PATTERN = re.compile(r"\b[1-9]\b")

# Load the JSON mapping (10-100) relative to where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
NUMBERS_JSON_PATH = SCRIPT_DIR / "number_mapping" / "numbers.json"

# Default location of the per-utterance 1-9 pronunciation manifest.
DEFAULT_DIGIT_MANIFEST_PATH = (
    SCRIPT_DIR.parent / "manifests" / "original_pronunciations.tsv"
)

try:
    with open(NUMBERS_JSON_PATH, "r", encoding="utf-8") as f:
        CEB_NUMBERS = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find {NUMBERS_JSON_PATH}", file=sys.stderr)
    print(
        "Please ensure numbers.json exists in the preprocessing/ directory.",
        file=sys.stderr,
    )
    sys.exit(1)

# Sanity check: numbers.json should only cover 10-100. Single digits are
# handled separately via the per-utterance manifest, so if numbers.json
# ever grows a "1".."9" key it would silently shadow the per-utterance
# lookup below. Warn loudly if that ever happens.
_overlapping_keys = sorted(
    k for k in CEB_NUMBERS if k.isdigit() and 1 <= int(k) <= 9
)
if _overlapping_keys:
    print(
        "WARNING: numbers.json contains single-digit keys"
        f" {_overlapping_keys}; these are expected to live in the"
        " per-utterance digit manifest instead and will be ignored in favor"
        " of it.",
        file=sys.stderr,
    )

# Pre-compile regex patterns for speed. Sort by integer value descending
# so '100' is matched before '10' and '1'. Single digits (1-9) are excluded
# here on purpose -- they are handled by the per-utterance manifest lookup.
COMPILED_NUMBERS = [
    (re.compile(rf"\b{num_str}\b"), word)
    for num_str, word in sorted(
        CEB_NUMBERS.items(),
        key=lambda item: int(item[0]),
        reverse=True,
    )
    if not (num_str.isdigit() and 1 <= int(num_str) <= 9)
]


# ---------------------------------------------------------------------------
# PER-UTTERANCE SINGLE-DIGIT (1-9) PRONUNCIATION MANIFEST
# ---------------------------------------------------------------------------


def load_digit_pronunciations(tsv_path: Path):
    """
    Load the per-utterance 1-9 pronunciation manifest.

    Expected TSV columns: speaker_id, wav_filename, transcript, saying
    (transcript is the raw digit "1".."9"; saying is the word actually
    spoken in that specific recording, e.g. "uno" or "usa" for "1").

    Returns:
        dict: wav_filename -> (transcript_digit, saying)
    """
    mapping = {}

    if not tsv_path.exists():
        print(
            f"WARNING: digit pronunciation manifest not found at {tsv_path};"
            " Random Digit single-digit (1-9) utterances will be left"
            " unexpanded.",
            file=sys.stderr,
        )
        return mapping

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_cols = {"wav_filename", "transcript", "saying"}
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            print(
                f"ERROR: digit pronunciation manifest {tsv_path} is missing "
                f"required column(s): {sorted(missing_cols)}",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            wav = (row.get("wav_filename") or "").strip()
            transcript = (row.get("transcript") or "").strip()
            saying = (row.get("saying") or "").strip()

            if not wav or not saying:
                continue

            if wav in mapping and mapping[wav] != (transcript, saying):
                print(
                    f"WARNING: duplicate wav_filename {wav!r} in digit "
                    "pronunciation manifest with a different entry "
                    f"({mapping[wav]!r} vs {(transcript, saying)!r}); "
                    "keeping the first one seen.",
                    file=sys.stderr,
                )
                continue

            mapping[wav] = (transcript, saying)

    return mapping


# ---------------------------------------------------------------------------
# SOURCE FILTERING
# ---------------------------------------------------------------------------


def should_exclude_source(source_file: str) -> bool:
    return any(
        source_file.startswith(prefix) for prefix in EXCLUDE_SOURCE_PREFIXES
    )


def is_digit_entry(source_file: str) -> bool:
    return source_file.strip().lower() == "random digit"


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------


def clean_text(
    raw: str, filename: str, source_file: str, digit_map: dict, stats: dict
) -> str:
    """
    Perform corpus-level transcript cleanup.

    filename    - the wav filename this transcript belongs to; used to
                  look up the correct 1-9 pronunciation for THIS
                  specific utterance.
    source_file - the transcript source (e.g. "Random Digit",
                  "Iso_...", "CEB_..."). Single-digit (1-9) expansion
                  via the per-utterance manifest is ONLY attempted
                  when this is a Random Digit entry.
    digit_map   - dict from load_digit_pronunciations(): wav_filename ->
                  (transcript_digit, saying)
    stats       - mutable dict for counters (mutated in place)
    """
    text = raw.strip()

    # Remove surrounding quotation marks if present.
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]

    # Remove parenthetical English glosses/translations.
    text = PAREN_PATTERN.sub("", text)

    # Expand single digits (1-9) using the per-utterance pronunciation
    # manifest — but ONLY for Random Digit source entries. A bare 1-9
    # digit appearing in some other kind of utterance is left as-is,
    # since the manifest was only built to cover Random Digit
    # recordings and doesn't necessarily have (or need) an entry for
    # every wav_filename in the corpus.
    if is_digit_entry(source_file):

        def _replace_single_digit(match: "re.Match") -> str:
            digit = match.group(0)
            entry = digit_map.get(filename)

            if entry is None:
                stats["single_digit_missing"] = (
                    stats.get("single_digit_missing", 0) + 1
                )
                print(
                    f"WARNING: no pronunciation entry for {filename} (digit"
                    f" {digit!r}, source=Random Digit); leaving digit"
                    " unexpanded.",
                    file=sys.stderr,
                )
                return digit

            expected_digit, saying = entry

            if expected_digit != digit:
                stats["single_digit_mismatch"] = (
                    stats.get("single_digit_mismatch", 0) + 1
                )
                print(
                    f"WARNING: digit mismatch for {filename}: transcript has "
                    f"{digit!r} but manifest says {expected_digit!r}; using "
                    f"manifest saying {saying!r} anyway.",
                    file=sys.stderr,
                )

            stats["single_digit_expanded"] = (
                stats.get("single_digit_expanded", 0) + 1
            )
            return saying

        text = SINGLE_DIGIT_PATTERN.sub(_replace_single_digit, text)

    else:
        # Not a Random Digit entry — if a bare 1-9 digit somehow shows
        # up here anyway, count it but don't touch it (no manifest
        # coverage is expected/guaranteed for non-Random-Digit sources).
        if SINGLE_DIGIT_PATTERN.search(text):
            stats["single_digit_skipped_non_random"] = (
                stats.get("single_digit_skipped_non_random", 0) + 1
            )

    # Expand multi-digit numbers (10-100) using the fixed corpus-wide
    # mapping. Applied regardless of source, since 10-100 pronunciation
    # doesn't vary by speaker the way 1-9 does.
    for pattern, word in COMPILED_NUMBERS:
        text = pattern.sub(word, text)

    # Collapse tabs / repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# LOG FILE
# ---------------------------------------------------------------------------


def find_log_file(speaker_dir: Path):
    """
    Find the .log file directly inside a speaker folder.
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
# TRANSCRIPT PARSING
# ---------------------------------------------------------------------------


def parse_transcript_lines(log_path: Path):
    """
    Yield: (wav_filename, source_file, raw_text)
    """
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

            filename, source_file, raw_text = match.groups()

            yield (filename, source_file, raw_text)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------


def log_exclusion(
    speaker_id: str, filename: str, source_file: str, reason: str
):
    print(
        f"EXCLUDED [{speaker_id}] "
        f"{filename} | "
        f"source={source_file!r} | "
        f"reason={reason}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract and clean corpus transcripts into one "
            ".txt file per speaker"
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Path to the corpus root folder",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output folder for speaker transcript files",
    )

    parser.add_argument(
        "--digit-map",
        default=str(DEFAULT_DIGIT_MANIFEST_PATH),
        help=(
            "Path to the TSV mapping each single-digit (1-9) Random "
            "Digit utterance's wav_filename to its actually-spoken word "
            "(columns: speaker_id, wav_filename, transcript, saying). "
            f"Default: {DEFAULT_DIGIT_MANIFEST_PATH}"
        ),
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    digit_map_path = Path(args.digit_map)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    digit_map = load_digit_pronunciations(digit_map_path)

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not speaker_dirs:
        print(
            f"ERROR: no speaker subfolders found under {root}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(speaker_dirs)} speaker folders under {root}")
    print(f"Output folder: {out_root.resolve()}")
    print(f"Excluded source prefixes: {EXCLUDE_SOURCE_PREFIXES}")
    print(f"Random Digit mode: {DIGIT_MODE!r}")
    print(
        "Loaded number expansion mapping (10-100) from:"
        f" {NUMBERS_JSON_PATH.name}"
    )
    print(
        "Loaded single-digit (1-9) pronunciation manifest from:"
        f" {digit_map_path} ({len(digit_map)} entries)"
    )
    print("-" * 60)

    total_written = 0
    total_excluded_source = 0
    total_excluded_digit = 0
    total_excluded_empty = 0
    digit_stats = {}

    missing_logs = []

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        log_file = find_log_file(speaker_dir)

        if not log_file:
            missing_logs.append(speaker_id)
            print(
                f"WARNING: [{speaker_id}] no .log file found", file=sys.stderr
            )
            continue

        out_path = out_root / f"{speaker_id}.txt"
        speaker_utterances = []

        for filename, source_file, raw_text in parse_transcript_lines(
            log_file
        ):

            if should_exclude_source(source_file):
                total_excluded_source += 1
                log_exclusion(
                    speaker_id, filename, source_file, "source prefix excluded"
                )
                continue

            if is_digit_entry(source_file) and DIGIT_MODE == "exclude":
                total_excluded_digit += 1
                log_exclusion(
                    speaker_id, filename, source_file, "Random Digit excluded"
                )
                continue

            cleaned = clean_text(
                raw_text, filename, source_file, digit_map, digit_stats
            )

            if not cleaned:
                total_excluded_empty += 1
                log_exclusion(
                    speaker_id, filename, source_file, "empty after cleaning"
                )
                continue

            speaker_utterances.append(f"{filename}\t{cleaned}")
            total_written += 1

        if speaker_utterances:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(speaker_utterances))
                f.write("\n")
            print(
                f"[{speaker_id}] wrote {len(speaker_utterances)} "
                f"utterances -> {out_path}"
            )

    if missing_logs:
        print(
            f"\nWARNING: {len(missing_logs)} speaker(s) had no .log file: "
            f"{missing_logs[:10]}{'...' if len(missing_logs) > 10 else ''}",
            file=sys.stderr,
        )

    print("-" * 60)
    print("Done.")
    print(f"Utterances written           : {total_written}")
    print(f"Excluded (source prefix)     : {total_excluded_source}")
    print(f"Excluded (digit mode)        : {total_excluded_digit}")
    print(f"Excluded (empty after clean) : {total_excluded_empty}")
    print(
        "Speakers processed           :"
        f" {len(speaker_dirs) - len(missing_logs)}"
    )
    print(f"Speakers missing .log        : {len(missing_logs)}")
    print("-" * 60)
    print("Single-digit (1-9) expansion (Random Digit sources only):")
    print(
        "  Expanded via manifest              :"
        f" {digit_stats.get('single_digit_expanded', 0)}"
    )
    print(
        "  Missing manifest entry             :"
        f" {digit_stats.get('single_digit_missing', 0)}"
    )
    print(
        "  Digit/manifest mismatch            :"
        f" {digit_stats.get('single_digit_mismatch', 0)}"
    )
    print(
        "  Bare digit in non-Random source    :"
        f" {digit_stats.get('single_digit_skipped_non_random', 0)} (left"
        " unexpanded)"
    )


if __name__ == "__main__":
    main()
