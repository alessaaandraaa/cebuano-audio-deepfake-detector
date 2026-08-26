"""
resample_dataset.py

Resamples/converts audio to 16kHz mono while excluding transcript sources
that are not wanted for the dataset.

Currently excludes:
    TGL_*

Raw originals are never modified.

Usage:
    python preprocessing\\resample_dataset.py --root "C:\\path\\to\\raw\\bonafide" --out "data\\processed\\bonafide"
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
    import librosa
except ImportError:
    print(
        "Missing dependency(s). Run: pip install soundfile librosa",
        file=sys.stderr,
    )
    sys.exit(1)


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}

TARGET_SR = 16000

# ---------------------------------------------------------------------------
# SOURCE FILTER
# ---------------------------------------------------------------------------

EXCLUDE_SOURCE_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)

# ---------------------------------------------------------------------------


def parse_transcript_sources(log_path: Path):
    """
    Returns:

        {
            "clip.wav": "TGL_spontaneous.txt",
            "other.wav": "CEB_common.txt",
        }
    """

    sources = {}

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

                parts = line.split('"')

                if len(parts) < 3:
                    continue

                filename = parts[0].strip()

                if not filename.endswith(".wav"):
                    continue

                source_file = parts[1].strip()

                sources[filename] = source_file

    except Exception as e:
        print(
            f"WARNING: failed to parse {log_path}: {e}",
            file=sys.stderr,
        )

    return sources


def find_log_file(speaker_dir: Path):

    log_files = list(
        speaker_dir.glob("*.log")
    )

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

    return any(
        source_file.startswith(prefix)
        for prefix in EXCLUDE_SOURCE_PREFIXES
    )


def build_exclusion_map(root: Path):
    """
    Reads every speaker's .log file and builds:

        {
            absolute_audio_path: source_file
        }

    for excluded clips.
    """

    excluded = {}

    speaker_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
    )

    for speaker_dir in speaker_dirs:

        log_file = find_log_file(speaker_dir)

        if not log_file:
            print(
                f"WARNING: [{speaker_dir.name}] "
                f"no .log file found",
                file=sys.stderr,
            )
            continue

        sources = parse_transcript_sources(
            log_file
        )

        for filename, source_file in sources.items():

            if not should_exclude_source(source_file):
                continue

            audio_path = speaker_dir / filename

            excluded[str(audio_path.resolve())] = source_file

    return excluded


def find_audio_files(root: Path):

    return [
        f
        for f in root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in AUDIO_EXTENSIONS
    ]


def resample_file(
    src: Path,
    dst: Path,
) -> bool:

    try:

        y, sr = librosa.load(
            str(src),
            sr=TARGET_SR,
            mono=True,
        )

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            str(dst),
            y,
            TARGET_SR,
        )

        return True

    except Exception as e:

        print(
            f"FAILED: {src} -> {e}",
            file=sys.stderr,
        )

        return False


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Resample dataset to 16kHz mono "
            "while excluding TGL sources"
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Path to raw audio root folder",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output folder for resampled audio",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    if not root.exists():
        print(
            f"ERROR: root path does not exist: {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # Build exclusion list BEFORE processing audio.
    # ---------------------------------------------------------------

    print("Reading .log files for transcript exclusions...")

    excluded = build_exclusion_map(root)

    print(
        f"Found {len(excluded)} excluded audio files."
    )

    print("\nALL EXCLUDED FILES:")

    for path, source_file in sorted(excluded.items()):
        f = Path(path)
        speaker_id = f.parent.name

        print(
            f"EXCLUDED [{speaker_id}] "
            f"{f.name} | source={source_file}",
        )

    # ---------------------------------------------------------------
    # Find audio
    # ---------------------------------------------------------------

    audio_files = find_audio_files(root)

    if not audio_files:
        print(
            f"No audio files found under {root}. Nothing to do.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Found {len(audio_files)} audio files under {root}"
    )

    print(
        f"Resampling to {TARGET_SR}Hz mono -> "
        f"{out_root}"
    )

    print("-" * 60)

    success_count = 0
    fail_count = 0
    excluded_count = 0

    for i, src in enumerate(
        audio_files,
        start=1,
    ):

        src_resolved = str(
            src.resolve()
        )

        # -----------------------------------------------------------
        # Exclude TGL
        # -----------------------------------------------------------

        if src_resolved in excluded:

            excluded_count += 1

            print(
                f"EXCLUDED [{src.parent.name}] "
                f"{src.name} | "
                f"source={excluded[src_resolved]}"
            )

            continue

        # -----------------------------------------------------------
        # Resample
        # -----------------------------------------------------------

        rel_path = src.relative_to(root)

        dst = out_root / rel_path

        if resample_file(src, dst):
            success_count += 1
        else:
            fail_count += 1

        if (
            i % 500 == 0
            or i == len(audio_files)
        ):
            print(
                f"  Processed {i}/{len(audio_files)} "
                f"({success_count} ok, "
                f"{fail_count} failed, "
                f"{excluded_count} excluded)"
            )

    print("-" * 60)

    print(
        f"Done. "
        f"{success_count} succeeded, "
        f"{fail_count} failed, "
        f"{excluded_count} excluded."
    )

    print(
        f"Resampled audio written to: "
        f"{out_root.resolve()}"
    )

    if fail_count > 0:
        print(
            f"\n{fail_count} file(s) failed to resample -- "
            f"check the FAILED lines above and decide "
            f"whether to fix/exclude them."
        )


if __name__ == "__main__":
    main()