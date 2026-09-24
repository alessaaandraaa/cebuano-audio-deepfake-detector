r"""
inventory_bonafide.py

Scans a bonafide audio dataset organized as:

    root/
      speaker_001/
        clip_001.wav
        clip_002.wav
        speaker_001.log
      speaker_002/
        ...

Uses each speaker's .log file to determine which audio clips should be
included.

TGL_* transcript sources are excluded automatically.

The manifest contains one row per INCLUDED audio clip and is labeled
label=bonafide automatically.

Usage:
    python preprocessing\inventory_bonafide.py \
        --root "C:\path\to\bonafide" \
        --out manifests\manifest_bonafide.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

try:
    import soundfile as sf
except ImportError:
    print("Missing dependency. Run: pip install soundfile", file=sys.stderr)
    sys.exit(1)


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# SOURCE FILTER
# ---------------------------------------------------------------------------

# Any transcript source beginning with one of these prefixes is excluded.
#
# Example:
#   "TGL_spontaneous.txt" -> excluded
#   "TGL_reading.txt"     -> excluded
#   "CEB_something.txt"   -> included
#   "Random Digit"        -> included
#
EXCLUDE_SOURCE_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)


# ---------------------------------------------------------------------------


LOG_FIELD_MAP = {
    "SessionID": "session_id",
    "SessionDate": "session_date",
    "SessionEnvironment": "session_environment",
    "SessionComments": "session_comments",
    "SpeakerID": "speaker_id_log",
    "SpeakerName": "speaker_name",
    "SpeakerAge": "speaker_age",
    "SpeakerGender": "speaker_gender",
    "SpeakerDialect": "speaker_dialect",
    "MotherDialect": "mother_dialect",
    "FatherDialect": "father_dialect",
    "SpeakerProfession": "speaker_profession",
    "SpeakerComments": "speaker_comments_log",
}


def parse_log_file(log_path: Path) -> tuple[dict, dict]:
    """
    Parses a session .log file.

    Returns:
        metadata:
            Session/speaker metadata.

        transcript_sources:
            Mapping:
                wav_filename -> source_file

            Example:
                {
                    "0205.111024.100407.0391.wav": "TGL_spontaneous.txt"
                }
    """

    metadata = {}
    transcript_sources = {}

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

                # -----------------------------------------------------------
                # Metadata
                # -----------------------------------------------------------

                if "=" in line and not line.startswith('"'):
                    key, _, value = line.partition("=")

                    key = key.strip()
                    value = value.strip().strip('"')

                    if key in LOG_FIELD_MAP:
                        metadata[LOG_FIELD_MAP[key]] = value

                    continue

                # -----------------------------------------------------------
                # Transcript line
                # -----------------------------------------------------------

                parts = line.split('"')

                # Expected:
                #
                # filename "source" "transcript"
                #
                # Example:
                # 0205....wav "TGL_spontaneous.txt" "Magbanggit..."
                #
                if len(parts) >= 3:
                    filename_part = parts[0].strip()

                    if filename_part.endswith(".wav"):
                        source_file = parts[1].strip()

                        transcript_sources[filename_part] = source_file

    except Exception as e:
        print(
            f"WARNING: failed to parse log file {log_path}: {e}",
            file=sys.stderr,
        )

    return metadata, transcript_sources


def find_log_file(speaker_dir: Path):
    """Finds a .log file directly inside a speaker folder."""

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


def should_exclude_source(source_file: str) -> bool:
    """Returns True if a transcript source should be excluded."""

    return any(
        source_file.startswith(prefix) for prefix in EXCLUDE_SOURCE_PREFIXES
    )


def scan_dataset(root: Path):
    rows = []
    errors = []
    missing_logs = []

    excluded_source_count = 0
    excluded_source_files = []

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not speaker_dirs:
        print(
            f"WARNING: no subfolders found directly under {root}.",
            file=sys.stderr,
        )

    for speaker_dir in speaker_dirs:

        speaker_id = speaker_dir.name

        log_file = find_log_file(speaker_dir)

        if not log_file:
            missing_logs.append(speaker_id)
            log_metadata = {}
            transcript_sources = {}
        else:
            log_metadata, transcript_sources = parse_log_file(log_file)

        logged_id = log_metadata.get("speaker_id_log")

        if logged_id and logged_id != speaker_id:
            print(
                f"NOTE: folder '{speaker_id}' != "
                f"log SpeakerID '{logged_id}' "
                "(using folder name as canonical id)",
                file=sys.stderr,
            )

        audio_files = [
            f
            for f in speaker_dir.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()
        ]

        for f in audio_files:

            # ---------------------------------------------------------------
            # SOURCE FILTER
            # ---------------------------------------------------------------

            source_file = transcript_sources.get(f.name)

            if source_file is None:
                print(
                    f"WARNING: [{speaker_id}] {f.name} "
                    "has no transcript source in .log",
                    file=sys.stderr,
                )

            elif should_exclude_source(source_file):

                excluded_source_count += 1

                excluded_source_files.append(
                    (
                        speaker_id,
                        f.name,
                        source_file,
                    )
                )

                print(
                    f"EXCLUDED [{speaker_id}] {f.name} | source={source_file}",
                )

                continue

            # ---------------------------------------------------------------
            # AUDIO VALIDATION
            # ---------------------------------------------------------------

            try:

                info = sf.info(str(f))
                duration = info.frames / info.samplerate

                row = {
                    "filepath": str(f.resolve()),
                    "filename": f.name,
                    "speaker_id": speaker_id,
                    "label": "bonafide",
                    "source": "bonafide",
                    "duration_sec": round(duration, 3),
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "transcript_source": source_file or "",
                }

                row.update(
                    {
                        k: log_metadata.get(k, "")
                        for k in LOG_FIELD_MAP.values()
                    }
                )

                rows.append(row)

            except Exception as e:
                errors.append(
                    (
                        str(f),
                        str(e),
                    )
                )

    if missing_logs:
        print(
            f"\nWARNING: {len(missing_logs)} speaker folder(s) "
            "had no .log file: "
            f"{missing_logs[:10]}"
            f"{'...' if len(missing_logs) > 10 else ''}",
            file=sys.stderr,
        )

    return (
        rows,
        errors,
        excluded_source_count,
        excluded_source_files,
    )


def write_manifest(rows, out_path: Path):

    if not rows:
        print(
            "No valid audio files found. Nothing written.",
            file=sys.stderr,
        )
        return

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with open(
        out_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows,
    errors,
    excluded_source_count,
    excluded_source_files,
    root: Path,
):

    speakers = sorted(set(r["speaker_id"] for r in rows))

    total_duration_hr = sum(r["duration_sec"] for r in rows) / 3600

    sample_rates = sorted(set(r["sample_rate"] for r in rows))

    channels = sorted(set(r["channels"] for r in rows))

    print()
    print("=" * 60)
    print(f"Bonafide dataset inventory: {root}")
    print("=" * 60)

    print(f"Total INCLUDED clips    : {len(rows)}")
    print(f"Total speakers          : {len(speakers)}")
    print(f"Total duration (hours)  : {total_duration_hr:.2f}")
    print(f"Excluded TGL clips      : {excluded_source_count}")
    print(f"Files that failed       : {len(errors)}")

    print(f"Sample rate(s) seen     : {sample_rates}")
    print(f"Channel count(s) seen   : {channels}")

    if len(sample_rates) > 1:
        print(
            "\n⚠ Multiple sample rates detected — "
            "resample to a consistent rate "
            "(likely 16kHz for XLS-R) before training."
        )

    if len(channels) > 1:
        print(
            "⚠ Mixed mono/stereo detected — convert to mono for consistency."
        )

    if excluded_source_files:
        print("\nTGL exclusions:")

        for speaker_id, filename, source_file in excluded_source_files[:10]:
            print(f"  [{speaker_id}] {filename} ({source_file})")

        if len(excluded_source_files) > 10:
            print(f"  ... and {len(excluded_source_files) - 10} more")

    if errors:
        print("\nFirst few failed files:")

        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    counts = Counter(r["speaker_id"] for r in rows)

    counts_sorted = sorted(
        counts.items(),
        key=lambda x: x[1],
    )

    print("\nSpeakers with fewest clips:")

    for spk, n in counts_sorted[:5]:
        print(f"  {spk}: {n} clips")

    print("Speakers with most clips:")

    for spk, n in counts_sorted[-5:]:
        print(f"  {spk}: {n} clips")

    speaker_gender = {}

    for r in rows:
        if r["speaker_id"] not in speaker_gender and r.get("speaker_gender"):
            speaker_gender[r["speaker_id"]] = r["speaker_gender"]

    gender_counts = Counter(speaker_gender.values())

    if gender_counts:
        print(f"\nSpeaker gender breakdown: {dict(gender_counts)}")

    speaker_dialect = {}

    for r in rows:
        if r["speaker_id"] not in speaker_dialect and r.get("speaker_dialect"):
            speaker_dialect[r["speaker_id"]] = r["speaker_dialect"]

    dialect_counts = Counter(speaker_dialect.values())

    if dialect_counts:
        print(f"Speaker dialect breakdown: {dict(dialect_counts)}")


def main():

    parser = argparse.ArgumentParser(
        description="Inventory bonafide audio dataset"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Path to bonafide root folder",
    )

    parser.add_argument(
        "--out",
        default="manifest_bonafide.csv",
        help="Output manifest CSV path",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(
            f"ERROR: root path does not exist: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    (
        rows,
        errors,
        excluded_source_count,
        excluded_source_files,
    ) = scan_dataset(root)

    write_manifest(
        rows,
        Path(args.out),
    )

    print_summary(
        rows,
        errors,
        excluded_source_count,
        excluded_source_files,
        root,
    )

    print(f"\nManifest written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
