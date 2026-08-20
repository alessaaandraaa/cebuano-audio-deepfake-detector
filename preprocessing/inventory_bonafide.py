"""
inventory_bonafide.py

Scans a bonafide audio dataset organized as:
    root/
      speaker_001/
        clip_001.wav
        clip_002.wav
        speaker_001.log
      speaker_002/
        ...

Validates each file, extracts speaker ID from the parent folder name,
parses the accompanying .log file for speaker/session metadata, and
writes a manifest CSV with one row per clip (this manifest is your
labeling step -- every row is tagged label=bonafide automatically).

Usage:
    python inventory_bonafide.py --root "C:\\path\\to\\bonafide" --out manifests\\manifest_bonafide.csv
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

# Fields expected from a session .log file, formatted as:  Key = "Value"
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


def parse_log_file(log_path: Path) -> dict:
    """Parses 'Key = Value' lines from a session .log file."""
    metadata = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                if key in LOG_FIELD_MAP:
                    metadata[LOG_FIELD_MAP[key]] = value
    except Exception as e:
        print(f"WARNING: failed to parse log file {log_path}: {e}", file=sys.stderr)
    return metadata


def find_log_file(speaker_dir: Path):
    """Finds a .log file directly inside a speaker folder (expects one)."""
    log_files = list(speaker_dir.glob("*.log"))
    if not log_files:
        return None
    if len(log_files) > 1:
        print(f"WARNING: multiple .log files in {speaker_dir}, using {log_files[0].name}",
              file=sys.stderr)
    return log_files[0]


def scan_dataset(root: Path):
    """Walks root/speaker_xxx/*.wav, merging in .log metadata per speaker."""
    rows, errors, missing_logs = [], [], []

    speaker_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not speaker_dirs:
        print(f"WARNING: no subfolders found directly under {root}.", file=sys.stderr)

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        log_file = find_log_file(speaker_dir)
        log_metadata = parse_log_file(log_file) if log_file else {}
        if not log_file:
            missing_logs.append(speaker_id)

        logged_id = log_metadata.get("speaker_id_log")
        if logged_id and logged_id != speaker_id:
            print(f"NOTE: folder '{speaker_id}' != log SpeakerID '{logged_id}' "
                  f"(using folder name as canonical id)", file=sys.stderr)

        audio_files = [
            f for f in speaker_dir.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()
        ]

        for f in audio_files:
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
                }
                row.update({k: log_metadata.get(k, "") for k in LOG_FIELD_MAP.values()})
                rows.append(row)
            except Exception as e:
                errors.append((str(f), str(e)))

    if missing_logs:
        print(f"\nWARNING: {len(missing_logs)} speaker folder(s) had no .log file: "
              f"{missing_logs[:10]}{'...' if len(missing_logs) > 10 else ''}", file=sys.stderr)

    return rows, errors


def write_manifest(rows, out_path: Path):
    if not rows:
        print("No valid audio files found. Nothing written.", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, errors, root: Path):
    speakers = sorted(set(r["speaker_id"] for r in rows))
    total_duration_hr = sum(r["duration_sec"] for r in rows) / 3600
    sample_rates = sorted(set(r["sample_rate"] for r in rows))
    channels = sorted(set(r["channels"] for r in rows))

    print("=" * 60)
    print(f"Bonafide dataset inventory: {root}")
    print("=" * 60)
    print(f"Total clips found      : {len(rows)}")
    print(f"Total speakers         : {len(speakers)}")
    print(f"Total duration (hours) : {total_duration_hr:.2f}")
    print(f"Sample rate(s) seen    : {sample_rates}")
    print(f"Channel count(s) seen  : {channels}")
    print(f"Files that failed      : {len(errors)}")

    if len(sample_rates) > 1:
        print("\n⚠ Multiple sample rates detected — resample to a consistent "
              "rate (likely 16kHz for XLS-R) before training.")
    if len(channels) > 1:
        print("⚠ Mixed mono/stereo detected — convert to mono for consistency.")

    if errors:
        print("\nFirst few failed files:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    counts = Counter(r["speaker_id"] for r in rows)
    counts_sorted = sorted(counts.items(), key=lambda x: x[1])
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
    parser = argparse.ArgumentParser(description="Inventory bonafide audio dataset")
    parser.add_argument("--root", required=True, help="Path to bonafide root folder")
    parser.add_argument("--out", default="manifest_bonafide.csv", help="Output manifest CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows, errors = scan_dataset(root)
    write_manifest(rows, Path(args.out))
    print_summary(rows, errors, root)
    print(f"\nManifest written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()