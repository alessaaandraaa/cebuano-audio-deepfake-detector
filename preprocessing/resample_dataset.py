"""
resample_dataset.py

Standalone audio standardization script. Walks a dataset organized as:
    root/
      speaker_001/
        clip_001.wav
        clip_002.wav
      speaker_002/
        ...

For every audio file found, resamples/converts it to 16kHz mono and
writes it to a mirrored folder structure under --out. Raw originals
under --root are never modified.

Usage:
    python preprocessing\resample_dataset.py --root "C:\\path\\to\\raw\\bonafide" --out data\\processed\\bonafide
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
    import librosa
except ImportError:
    print("Missing dependency(s). Run: pip install soundfile librosa", file=sys.stderr)
    sys.exit(1)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TARGET_SR = 16000


def find_audio_files(root: Path):
    """Recursively finds all audio files under root, skipping non-audio files."""
    return [
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]


def resample_file(src: Path, dst: Path) -> bool:
    """Loads src, resamples to TARGET_SR mono, writes to dst. Returns True on success."""
    try:
        y, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, TARGET_SR)
        return True
    except Exception as e:
        print(f"FAILED: {src} -> {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Resample a dataset to 16kHz mono")
    parser.add_argument("--root", required=True, help="Path to raw audio root folder")
    parser.add_argument("--out", required=True,
                         help="Output folder for resampled audio (mirrors root's structure)")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    audio_files = find_audio_files(root)
    if not audio_files:
        print(f"No audio files found under {root}. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files under {root}")
    print(f"Resampling to {TARGET_SR}Hz mono -> {out_root}")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for i, src in enumerate(audio_files, start=1):
        # preserve the same relative path/filename under out_root
        rel_path = src.relative_to(root)
        dst = out_root / rel_path

        if resample_file(src, dst):
            success_count += 1
        else:
            fail_count += 1

        if i % 500 == 0 or i == len(audio_files):
            print(f"  Processed {i}/{len(audio_files)} "
                  f"({success_count} ok, {fail_count} failed)")

    print("-" * 60)
    print(f"Done. {success_count} succeeded, {fail_count} failed.")
    print(f"Resampled audio written to: {out_root.resolve()}")

    if fail_count > 0:
        print(f"\n{fail_count} file(s) failed to resample -- check the FAILED lines "
              f"above and decide whether to fix/exclude them.")


if __name__ == "__main__":
    main()