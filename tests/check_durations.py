"""
check_durations.py

Loops through each speaker subfolder under a root folder
(e.g. data/processed/bonafide/0200, 0201, ...) and reports
duration stats per speaker folder and overall.

No log files, no manifests — just reads the audio files directly.

Usage:
    python check_durations.py --root "data\\processed\\bonafide"
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Missing dependency. Run: pip install soundfile", file=sys.stderr)
    sys.exit(1)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def bucket_counts(durations):
    n = len(durations)
    return {
        "<2s": sum(1 for d in durations if d < 2.0),
        "2-3s": sum(1 for d in durations if 2.0 <= d < 3.0),
        "3-8s": sum(1 for d in durations if 3.0 <= d < 8.0),
        ">=8s": sum(1 for d in durations if d >= 8.0),
    }, n


def main():
    parser = argparse.ArgumentParser(description="Check audio durations per speaker folder")
    parser.add_argument("--root", required=True, help="Root folder containing speaker subfolders (e.g. data/processed/bonafide)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not speaker_dirs:
        print(f"ERROR: no speaker subfolders found under {root}", file=sys.stderr)
        sys.exit(1)

    all_durations = []
    all_under_1s = []
    errors = []

    print("=" * 60)
    print(f"Root: {root.resolve()}")
    print(f"Speaker folders found: {len(speaker_dirs)}")
    print("=" * 60)

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        speaker_durations = []

        for f in sorted(speaker_dir.rglob("*")):
            if f.suffix.lower() not in AUDIO_EXTENSIONS or not f.is_file():
                continue

            try:
                info = sf.info(str(f))
                d = info.frames / info.samplerate
                speaker_durations.append(d)
                all_durations.append(d)

                if d < 1.0:
                    all_under_1s.append((str(f), d))

            except Exception as e:
                errors.append((str(f), str(e)))

        if speaker_durations:
            buckets, n = bucket_counts(speaker_durations)
            mean_d = sum(speaker_durations) / n
            under_1s_pct = 100 * buckets["<2s"] / n
            print(
                f"[{speaker_id}] {n:5d} clips | mean {mean_d:.2f}s | "
                f"<2s: {buckets['<2s']:4d} ({under_1s_pct:.1f}%)"
            )
        else:
            print(f"[{speaker_id}] no audio files found")

    if not all_durations:
        print("\nNo audio files found anywhere.", file=sys.stderr)
        sys.exit(1)

    n = len(all_durations)
    buckets, _ = bucket_counts(all_durations)

    print()
    print("=" * 60)
    print(f"OVERALL ({n} clips across {len(speaker_dirs)} speaker folders)")
    print("=" * 60)
    print(f"Mean duration : {sum(all_durations)/n:.2f}s")
    print(f"Min / Max     : {min(all_durations):.2f}s / {max(all_durations):.2f}s")
    print()
    print("Duration buckets:")
    for label, count in buckets.items():
        pct = 100 * count / n
        print(f"  {label:6s} : {count:6d}  ({pct:.1f}%)")

    if all_under_1s:
        print(f"\nFiles under 1s ({len(all_under_1s)} total, showing first 15):")
        for path, d in all_under_1s[:15]:
            print(f"  {d:.2f}s  {path}")
        if len(all_under_1s) > 15:
            print(f"  ... and {len(all_under_1s) - 15} more")

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:", file=sys.stderr)
        for path, err in errors[:5]:
            print(f"  {path}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()