"""
check_leading_silence.py

Checks leading silence duration for audio files in a folder (recursive).
Run separately on your bonafide root and your spoof root, then compare.

Usage:
    python tests/check_leading_silence.py --root "data\\processed\\bonafide"
    python tests/check_leading_silence.py --root "data\\processed\\meta-mms"
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Missing dependency. Run: pip install soundfile numpy", file=sys.stderr)
    sys.exit(1)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def leading_silence_ms(path, threshold=0.01):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return None  # entire file is below threshold
    return (above[0] / sr) * 1000


def main():
    parser = argparse.ArgumentParser(description="Check leading silence duration in audio files")
    parser.add_argument("--root", required=True, help="Folder to scan (recursive)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Amplitude threshold for 'silence' (default 0.01)")
    parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample this many files instead of scanning all")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    files = [f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()]

    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    if args.sample > 0 and args.sample < len(files):
        import random
        files = random.sample(files, args.sample)

    values = []
    silent_files = []

    for f in sorted(files):
        try:
            ms = leading_silence_ms(f, args.threshold)
            if ms is None:
                silent_files.append(str(f))
                continue
            values.append(ms)
        except Exception as e:
            print(f"WARNING: failed to read {f}: {e}", file=sys.stderr)

    if not values:
        print("No valid measurements.", file=sys.stderr)
        sys.exit(1)

    values_arr = np.array(values)

    print("=" * 60)
    print(f"Root: {root.resolve()}")
    print(f"Files measured: {len(values)}")
    print("=" * 60)
    print(f"Mean leading silence : {values_arr.mean():.2f} ms")
    print(f"Median               : {np.median(values_arr):.2f} ms")
    print(f"Min / Max            : {values_arr.min():.2f} ms / {values_arr.max():.2f} ms")
    print(f"Std dev              : {values_arr.std():.2f} ms")

    near_zero = np.sum(values_arr < 5)
    print(f"\nFiles with <5ms leading silence: {near_zero} ({100*near_zero/len(values):.1f}%)")

    if silent_files:
        print(f"\n{len(silent_files)} file(s) were entirely below threshold (fully silent or very quiet).")


if __name__ == "__main__":
    main()