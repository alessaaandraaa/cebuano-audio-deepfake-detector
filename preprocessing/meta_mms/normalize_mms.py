"""
normalize_mms.py

Normalizes ONLY the speakers selected for MMS-TTS.

Speaker selection comes from:

    manifests/mms_selected_speakers.csv

Expected manifest format:

    speaker_id,gender

Example:

    0213,female
    0214,male
    0215,male

Input transcripts:

    transcripts/
      0213.txt
      0214.txt
      0215.txt
      ...

Output:

    transcripts_mms/
      0213.txt
      0214.txt
      0215.txt
      ...

Input format:

    WAV_FILENAME<TAB>TRANSCRIPT

Output format:

    WAV_FILENAME<TAB>NORMALIZED_TRANSCRIPT


MMS-specific transformations:

  - Applies phonetic respelling for English proper nouns
    using ceb_english_mapping.json.
  - Removes Unicode diacritics.
  - Converts text to lowercase.
  - Removes punctuation.
  - Applies acronym fallback for uppercase acronyms.
  - Converts "&" to "ug".
  - Collapses whitespace.


IMPORTANT:

    This script does NOT decide which corpus utterances should
    exist.

    Speaker selection is determined exclusively by:

        manifests/mms_selected_speakers.csv

    The original transcript files are never modified.

    The English proper-noun mapping is applied BEFORE lowercase,
    diacritic removal, and punctuation removal so that the
    original transcript can match the mapping correctly.


Usage:

    python preprocessing/meta_mms/normalize_mms.py --root transcripts --out transcripts_mms
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# DEFAULT PATHS
# ---------------------------------------------------------------------------

# Project structure assumed:
#
#   project/
#   ├── manifests/
#   │   └── mms_selected_speakers.csv
#   │
#   └── preprocessing/
#       ├── ceb_english_mapping.json
#       │
#       └── meta_mms/
#           └── normalize_mms.py
#
# Therefore:
#
#   parents[0] = preprocessing/meta_mms
#   parents[1] = preprocessing
#   parents[2] = project root

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "manifests"
    / "mms_selected_speakers.csv"
)

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[1]
    / "ceb_english_mapping.json"
)


# ---------------------------------------------------------------------------
# PHONETIC MAPPING
# ---------------------------------------------------------------------------

def load_phonetic_map(
    mapping_path: Path,
) -> dict:
    """
    Load the English proper-noun phonetic mapping.

    The mapping is expected to be a JSON object:

        {
            "English phrase": "Cebuano phonetic spelling",
            ...
        }
    """

    if not mapping_path.exists():

        print(
            f"ERROR: phonetic mapping not found:\n"
            f"  {mapping_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        with open(
            mapping_path,
            "r",
            encoding="utf-8",
        ) as f:

            mapping = json.load(f)

    except Exception as e:

        print(
            f"ERROR: failed to load phonetic mapping:\n"
            f"  {mapping_path}\n"
            f"  {e}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not isinstance(mapping, dict):

        print(
            "ERROR: phonetic mapping must be a JSON object.",
            file=sys.stderr,
        )

        sys.exit(1)

    return mapping


# ---------------------------------------------------------------------------
# PHONETIC MAP LOOKUP
# ---------------------------------------------------------------------------

def build_normalized_mapping(
    phonetic_map: dict,
) -> dict:
    """
    Build a secondary lookup table for the mapping.

    This allows matching to survive harmless differences such as:

        leading/trailing whitespace
        repeated whitespace
        Unicode normalization differences

    IMPORTANT:

        The actual mapping values are NOT modified here.
    """

    normalized_map = {}

    for key, value in phonetic_map.items():

        if not isinstance(key, str):
            continue

        if not isinstance(value, str):
            continue

        normalized_key = normalize_lookup_key(
            key
        )

        if normalized_key:
            normalized_map[normalized_key] = value

    return normalized_map


def normalize_lookup_key(
    text: str,
) -> str:
    """
    Normalize ONLY for dictionary lookup.

    This is deliberately different from MMS normalization.

    We do NOT lowercase here because mapping keys may intentionally
    distinguish capitalization.
    """

    text = text.strip()

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def lookup_phonetic_mapping(
    text: str,
    phonetic_map: dict,
    normalized_map: dict,
):
    """
    Attempt to find a phonetic replacement.

    Lookup order:

        1. Exact original key.
        2. Whitespace/Unicode-normalized key.
        3. Case-insensitive normalized key.

    Returns:

        mapped text, or None if no mapping exists.
    """

    # --------------------------------------------------------------
    # Exact match
    # --------------------------------------------------------------

    if text in phonetic_map:

        value = phonetic_map[text]

        if isinstance(value, str) and value.strip():

            return value

    # --------------------------------------------------------------
    # Normalized match
    # --------------------------------------------------------------

    normalized_key = normalize_lookup_key(
        text
    )

    if normalized_key in normalized_map:

        value = normalized_map[
            normalized_key
        ]

        if isinstance(value, str) and value.strip():

            return value

    # --------------------------------------------------------------
    # Case-insensitive fallback
    # --------------------------------------------------------------

    lowered_key = normalized_key.casefold()

    for key, value in normalized_map.items():

        if key.casefold() == lowered_key:

            if (
                isinstance(value, str)
                and value.strip()
            ):

                return value

    return None


# ---------------------------------------------------------------------------
# MMS SPEAKER MANIFEST
# ---------------------------------------------------------------------------

def load_selected_speakers(
    manifest_path: Path,
) -> list[str]:
    """
    Load speaker IDs from the MMS speaker-selection manifest.

    Expected:

        speaker_id,gender

        0213,female
        0214,male
        0215,male

    Only the first column is used.
    """

    if not manifest_path.exists():

        print(
            f"ERROR: MMS speaker manifest not found:\n"
            f"  {manifest_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    speakers = []

    try:

        with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.reader(f)

            for line_number, row in enumerate(
                reader,
                start=1,
            ):

                if not row:
                    continue

                speaker_id = row[0].strip()

                if not speaker_id:
                    continue

                # Optional header.
                if (
                    line_number == 1
                    and speaker_id.lower()
                    in {
                        "speaker_id",
                        "speaker",
                        "id",
                    }
                ):
                    continue

                speakers.append(
                    speaker_id
                )

    except Exception as e:

        print(
            f"ERROR: failed to read MMS speaker "
            f"manifest:\n"
            f"  {manifest_path}\n"
            f"  {e}",
            file=sys.stderr,
        )

        sys.exit(1)

    # Remove duplicates while preserving order.
    speakers = list(
        dict.fromkeys(
            speakers
        )
    )

    if not speakers:

        print(
            f"ERROR: no speakers found in:\n"
            f"  {manifest_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    return speakers


# ---------------------------------------------------------------------------
# DIACRITIC REMOVAL
# ---------------------------------------------------------------------------

def remove_diacritics(
    text: str,
) -> str:
    """
    Remove Unicode diacritical marks.

    Examples:

        á -> a
        é -> e
        í -> i
        ó -> o
        ú -> u

    Example:

        ayáw na págtagád

    becomes:

        ayaw na pagtagad
    """

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    return "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )


# ---------------------------------------------------------------------------
# MMS NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_mms_text(
    text: str,
    phonetic_map: dict,
    normalized_map: dict,
) -> tuple[str, bool]:
    """
    Normalize one transcript for MMS.

    Returns:

        (normalized_text, mapping_applied)
    """

    text = text.strip()

    # --------------------------------------------------------------
    # 1. PHONETIC MAPPING
    # --------------------------------------------------------------
    #
    # This MUST happen BEFORE:
    #
    #   - lowercase
    #   - diacritic removal
    #   - punctuation removal
    #
    # because those operations can destroy the exact form needed
    # to identify an English proper noun.
    # --------------------------------------------------------------

    mapped_text = lookup_phonetic_mapping(
        text=text,
        phonetic_map=phonetic_map,
        normalized_map=normalized_map,
    )

    mapping_applied = (
        mapped_text is not None
    )

    if mapped_text is not None:

        text = mapped_text

    else:

        # ----------------------------------------------------------
        # Ampersand
        # ----------------------------------------------------------

        text = text.replace(
            "&",
            "ug",
        )

    # --------------------------------------------------------------
    # 2. REMOVE DIACRITICS
    # --------------------------------------------------------------

    text = remove_diacritics(
        text
    )

    # --------------------------------------------------------------
    # 3. LOWERCASE
    # --------------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------------
    # 4. REMOVE PUNCTUATION
    # --------------------------------------------------------------

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # --------------------------------------------------------------
    # 5. COLLAPSE WHITESPACE
    # --------------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return (
        text,
        mapping_applied,
    )


# ---------------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------------

def normalize_file(
    input_path: Path,
    output_path: Path,
    phonetic_map: dict,
    normalized_map: dict,
):
    """
    Normalize one speaker transcript file.

    Returns:

        (
            utterance_count,
            mapping_count
        )
    """

    normalized_lines = []

    mapping_count = 0

    with open(
        input_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.rstrip(
                "\n\r"
            )

            if not line.strip():
                continue

            # ------------------------------------------------------
            # Expected:
            #
            # filename.wav<TAB>transcript
            # ------------------------------------------------------

            if "\t" not in line:

                print(
                    f"WARNING: {input_path.name}:"
                    f"{line_number}: "
                    f"no TAB separator; "
                    f"skipping line",
                    file=sys.stderr,
                )

                continue

            filename, text = line.split(
                "\t",
                1,
            )

            filename = filename.strip()

            if not filename:

                print(
                    f"WARNING: {input_path.name}:"
                    f"{line_number}: "
                    f"empty filename; "
                    f"skipping line",
                    file=sys.stderr,
                )

                continue

            normalized, mapping_applied = (
                normalize_mms_text(
                    text=text,
                    phonetic_map=phonetic_map,
                    normalized_map=normalized_map,
                )
            )

            if mapping_applied:

                mapping_count += 1

                print(
                    f"  MAPPING [{filename}]"
                    f"\n"
                    f"    Original : {text.strip()}"
                    f"\n"
                    f"    Result   : {normalized}"
                )

            if not normalized:

                print(
                    f"WARNING: {input_path.name}:"
                    f"{line_number}: "
                    f"text became empty "
                    f"after MMS normalization",
                    file=sys.stderr,
                )

                continue

            normalized_lines.append(
                f"{filename}\t{normalized}"
            )

    if not normalized_lines:

        return (
            0,
            mapping_count,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                normalized_lines
            )
        )

        f.write("\n")

    return (
        len(normalized_lines),
        mapping_count,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize only MMS-selected "
            "speaker transcripts."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Input folder containing one .txt "
            "file per speaker."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output folder for MMS-normalized "
            "transcripts."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
        help=(
            "CSV containing the selected MMS "
            "speakers."
        ),
    )

    parser.add_argument(
        "--mapping",
        default=str(
            DEFAULT_MAPPING_PATH
        ),
        help=(
            "Path to ceb_english_mapping.json."
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    out_root = Path(
        args.out
    )

    manifest_path = Path(
        args.manifest
    )

    mapping_path = Path(
        args.mapping
    )

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if not root.exists():

        print(
            f"ERROR: input folder not found: "
            f"{root}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not root.is_dir():

        print(
            f"ERROR: input path is not a directory: "
            f"{root}",
            file=sys.stderr,
        )

        sys.exit(1)

    # ------------------------------------------------------------------
    # Load MMS speakers
    # ------------------------------------------------------------------

    selected_speakers = (
        load_selected_speakers(
            manifest_path
        )
    )

    # ------------------------------------------------------------------
    # Load mapping
    # ------------------------------------------------------------------

    phonetic_map = load_phonetic_map(
        mapping_path
    )

    normalized_map = (
        build_normalized_mapping(
            phonetic_map
        )
    )

    # ------------------------------------------------------------------
    # Find selected speaker transcripts
    # ------------------------------------------------------------------

    selected_files = []

    missing_speakers = []

    for speaker_id in selected_speakers:

        input_path = (
            root
            / f"{speaker_id}.txt"
        )

        if not input_path.exists():

            missing_speakers.append(
                speaker_id
            )

            continue

        if not input_path.is_file():

            missing_speakers.append(
                speaker_id
            )

            continue

        selected_files.append(
            input_path
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(
        f"Input folder       : "
        f"{root.resolve()}"
    )

    print(
        f"Output folder      : "
        f"{out_root.resolve()}"
    )

    print(
        f"MMS speaker CSV    : "
        f"{manifest_path.resolve()}"
    )

    print(
        f"Phonetic mapping   : "
        f"{mapping_path.resolve()}"
    )

    print(
        f"Mapping entries    : "
        f"{len(phonetic_map)}"
    )

    print(
        f"Speakers selected  : "
        f"{len(selected_speakers)}"
    )

    print(
        f"Transcripts found  : "
        f"{len(selected_files)}"
    )

    print("-" * 60)

    # ------------------------------------------------------------------
    # Warn about missing transcripts
    # ------------------------------------------------------------------

    if missing_speakers:

        print(
            "WARNING: selected MMS speakers "
            "without transcripts:",
            file=sys.stderr,
        )

        for speaker_id in missing_speakers:

            print(
                f"  - {speaker_id}",
                file=sys.stderr,
            )

        print(
            "-" * 60
        )

    # ------------------------------------------------------------------
    # Process ONLY selected speakers
    # ------------------------------------------------------------------

    total_utterances = 0
    total_mappings = 0

    for input_path in selected_files:

        output_path = (
            out_root
            / input_path.name
        )

        (
            count,
            mapping_count,
        ) = normalize_file(
            input_path=input_path,
            output_path=output_path,
            phonetic_map=phonetic_map,
            normalized_map=normalized_map,
        )

        total_utterances += count
        total_mappings += mapping_count

        print(
            f"[{input_path.stem}] "
            f"normalized {count} utterances "
            f"({mapping_count} mapped) "
            f"-> {output_path}"
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    print("-" * 60)

    print("Done.")

    print(
        f"Speakers selected        : "
        f"{len(selected_speakers)}"
    )

    print(
        f"Speaker files processed  : "
        f"{len(selected_files)}"
    )

    print(
        f"Missing transcripts      : "
        f"{len(missing_speakers)}"
    )

    print(
        f"Utterances normalized    : "
        f"{total_utterances}"
    )

    print(
        f"Phonetic mappings applied: "
        f"{total_mappings}"
    )

    print(
        f"Output folder            : "
        f"{out_root.resolve()}"
    )


if __name__ == "__main__":
    main()