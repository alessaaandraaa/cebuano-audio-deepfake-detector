"""
extract_transcripts.py

Walks EVERY speaker folder under --root, parses each speaker's .log file,
cleans each utterance's transcript text for TTS input, and writes ONE .txt
file per speaker.

Outputs have been modified for MMS-TTS compatibility:
  - Applies phonetic respelling for English proper nouns via ceb_english_mapping.json
  - Converts all text to lowercase
  - Strips punctuation (.,-) to prevent tokenizer crashes

Output structure:
    root/0204/0204.log
    -> out/0204.txt

Each line in the output contains:
    WAV_FILENAME<TAB>CLEANED_TRANSCRIPT

python preprocessing\extract_transcripts.py --root "C:\Users\Ninzz\Programming\PLD\up-dsp-pld\PLD\CEB" --out transcripts
"""

import argparse
import re
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# PHONETIC MAPPING FOR MMS-TTS
# ---------------------------------------------------------------------------
# Loads the generated Bisaya phonetic map for English proper nouns.
PHONETIC_MAP = {}
try:
    with open("preprocessing/ceb_english_mapping.json", "r", encoding="utf-8") as f:
        PHONETIC_MAP = json.load(f)
except FileNotFoundError:
    print("WARNING: ceb_english_mapping.json not found. Skipping phonetic dictionary replacements.", file=sys.stderr)

# ---------------------------------------------------------------------------
# TOGGLES
# ---------------------------------------------------------------------------

EXCLUDE_SOURCE_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)

DIGIT_MODE = "keep"

# ---------------------------------------------------------------------------
# REGEX
# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(
    r'^(\S+\.wav)\s+"([^"]*)"\s+(.*)$'
)

PAREN_PATTERN = re.compile(
    r'\s*\([^)]*\)'
)

# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(raw: str, source_file: str) -> str:
    """
    Cleans a raw transcript and applies phonetic replacements.
    """
    text = raw.strip()

    # 1. Remove surrounding quotation marks.
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]

    # 2. Remove parenthetical translations/glosses.
    text = PAREN_PATTERN.sub("", text)

    # 3. Apply Phonetic Replacements for Iso Prompts
    if "Iso_" in source_file:
        if text in PHONETIC_MAP and PHONETIC_MAP[text]:
            text = PHONETIC_MAP[text]
        else:
            # Fallback: space out acronyms so MMS doesn't garble them (e.g., "MIA" -> "m i a")
            text = re.sub(r"\b([A-Z]{2,})\b", lambda m: " ".join(list(m.group(1).lower())), text)
            text = text.replace("&", "ug")

    # 4. MMS-TTS strict cleaning
    # Model requires lowercase and fails on most punctuation.
    text = text.lower()

    # 5. Collapse tabs / repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ---------------------------------------------------------------------------
# LOG FILE
# ---------------------------------------------------------------------------

def find_log_file(speaker_dir: Path):
    """Find the .log file directly inside a speaker folder."""
    log_files = list(speaker_dir.glob("*.log"))

    if not log_files:
        return None

    if len(log_files) > 1:
        print(
            f"WARNING: multiple .log files in {speaker_dir}, using {log_files[0].name}",
            file=sys.stderr,
        )

    return log_files[0]

# ---------------------------------------------------------------------------
# TRANSCRIPT PARSING
# ---------------------------------------------------------------------------

def parse_transcript_lines(log_path: Path):
    """Yields (source_wav_filename, source_file, raw_text) for each transcript line."""
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
            yield (filename, source_file, rest)

# ---------------------------------------------------------------------------
# SOURCE FILTERING
# ---------------------------------------------------------------------------

def should_exclude_source(source_file: str) -> bool:
    """Returns True if the transcript source should be excluded."""
    return any(source_file.startswith(prefix) for prefix in EXCLUDE_SOURCE_PREFIXES)

def is_digit_entry(source_file: str) -> bool:
    """Returns True if the source is Random Digit."""
    return source_file.strip().lower() == "random digit"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def log_exclusion(speaker_id: str, filename: str, source_file: str, reason: str):
    print(f"EXCLUDED [{speaker_id}] {filename} | source={source_file!r} | reason={reason}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract + clean utterances into one .txt file per speaker"
    )

    parser.add_argument("--root", required=True, help="Path to bonafide root folder")
    parser.add_argument("--out", required=True, help="Output folder for speaker transcript files")
    
    args = parser.parse_args()
    root = Path(args.root)
    out_root = Path(args.out)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not speaker_dirs:
        print(f"ERROR: no speaker subfolders found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Output folder: {out_root.resolve()}")
    print(f"Found {len(speaker_dirs)} speaker folders under {root}")
    print("-" * 60)

    total_written = 0
    total_excluded_source = 0
    total_excluded_digit = 0
    total_excluded_empty = 0
    missing_logs = []

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        log_file = find_log_file(speaker_dir)

        if not log_file:
            missing_logs.append(speaker_id)
            print(f"WARNING: [{speaker_id}] no .log file found", file=sys.stderr)
            continue

        out_path = out_root / f"{speaker_id}.txt"
        speaker_utterances = []

        for filename, source_file, raw_text in parse_transcript_lines(log_file):

            if should_exclude_source(source_file):
                total_excluded_source += 1
                log_exclusion(speaker_id, filename, source_file, "source prefix excluded")
                continue

            if is_digit_entry(source_file) and DIGIT_MODE == "exclude":
                total_excluded_digit += 1
                log_exclusion(speaker_id, filename, source_file, "Random Digit excluded")
                continue

            # CLEAN TRANSCRIPT WITH SOURCE FILE PASSED IN
            cleaned = clean_text(raw_text, source_file)

            if not cleaned:
                total_excluded_empty += 1
                log_exclusion(speaker_id, filename, source_file, "empty after cleaning")
                continue

            speaker_utterances.append(f"{filename}\t{cleaned}")
            total_written += 1

        if speaker_utterances:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(speaker_utterances))
                f.write("\n")
            print(f"[{speaker_id}] wrote {len(speaker_utterances)} utterances -> {out_path}")

    if missing_logs:
        print(
            f"WARNING: {len(missing_logs)} speaker(s) had no .log file: {missing_logs[:10]}...",
            file=sys.stderr,
        )

    print("-" * 60)
    print("Done.")
    print(f"Utterances written          : {total_written}")
    print(f"Excluded (source prefix)    : {total_excluded_source}")
    print(f"Excluded (digit mode)       : {total_excluded_digit}")
    print(f"Excluded (empty after clean): {total_excluded_empty}")
    print(f"Speakers processed          : {len(speaker_dirs) - len(missing_logs)}")

if __name__ == "__main__":
    main()