"""
extract_english_entities.py

Walks EVERY speaker folder under --root, parses each speaker's .log file,
and extracts every UNIQUE utterance from the specified English-heavy 
isolated prompt lists.

Outputs a JSON template file where you can monitor all the English terms
and eventually fill in their phonetic Bisaya spellings.

Usage:
    python preprocessing\extract_english_entities.py --root "C:\path\to\bonafide" --out "english_words_to_map.json"
"""

import argparse
import re
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# TARGET FILES
# ---------------------------------------------------------------------------
# Add or remove any source files here that you want to monitor
TARGET_SOURCE_FILES = {
    "CEB_Iso_Airlines.txt",
    "CEB_Iso_Cities.txt",
    "CEB_Iso_Companies.txt",
    "CEB_Iso_Hotels.txt",
    "CEB_Iso_Landmarks.txt",
    "CEB_Iso_NamesMale.txt",
    "CEB_Iso_NamesFem.txt",
    "CEB_Iso_Surnames_Countries.txt"
}

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
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Basic cleaning to get just the text."""
    text = raw.strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    
    text = PAREN_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_log_file(speaker_dir: Path):
    """Find the .log file directly inside a speaker folder."""
    log_files = list(speaker_dir.glob("*.log"))
    if not log_files:
        return None
    if len(log_files) > 1:
        print(f"WARNING: multiple .log files in {speaker_dir}, using {log_files[0].name}", file=sys.stderr)
    return log_files[0]


def parse_transcript_lines(log_path: Path):
    """Yields (filename, source_file, raw_text) for each transcript line."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "=" in line and not line.strip().startswith('"'):
                continue
                
            m = LINE_PATTERN.match(line.strip())
            if not m:
                continue
                
            yield m.groups()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique English utterances from specific Iso prompt files."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to bonafide root folder",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file name (e.g., english_map.json)",
    )

    args = parser.parse_args()
    root = Path(args.root)
    out_file = Path(args.out)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"Scanning {len(speaker_dirs)} speaker folders for English prompts...")

    unique_prompts = set()
    total_found = 0

    for speaker_dir in speaker_dirs:
        log_file = find_log_file(speaker_dir)
        if not log_file:
            continue

        for filename, source_file, raw_text in parse_transcript_lines(log_file):
            if source_file in TARGET_SOURCE_FILES:
                cleaned = clean_text(raw_text)
                if cleaned:
                    unique_prompts.add(cleaned)
                    total_found += 1

    print("-" * 60)
    print("Done scanning.")
    print(f"Total English utterances found   : {total_found}")
    print(f"Unique English utterances to map : {len(unique_prompts)}")
    
    # Save to JSON
    mapping_dict = {prompt: "" for prompt in sorted(unique_prompts)}
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=4, ensure_ascii=False)

    print(f"\nWrote template mapping dictionary to: {out_file.resolve()}")


if __name__ == "__main__":
    main()