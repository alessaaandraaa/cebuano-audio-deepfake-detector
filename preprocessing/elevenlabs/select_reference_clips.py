r"""
select_reference_clips.py

For each speaker listed in elevenlabs_selected_speakers.csv, scans that
speaker's bonafide audio folder, randomly selects 3-5 utterances that are
at least --min-duration seconds long, and MOVES them into:

    <out>/<speaker_id>/<filename>.wav

Usage:
    python select_reference_clips.py ^
        --csv manifests\elevenlabs_selected_speakers.csv ^
        --audio-root data\processed\bonafide ^
        --out elevenlabs-reference

Options:
    --min-duration   Minimum clip duration in seconds (default 10.0)
    --min-per-speaker / --max-per-speaker
                     How many clips to select (default 3 / 5)
    --seed           Random seed for reproducibility (default 42)
    --dry-run        Print what would be moved without actually moving anything
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Missing dependency. Run: pip install soundfile", file=sys.stderr)
    sys.exit(1)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_speaker_ids(csv_path: Path):
    speakers = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "speaker_id" not in reader.fieldnames:
            print(
                "ERROR: CSV must contain a 'speaker_id' column.",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            sid = row["speaker_id"].strip()
            if sid:
                speakers.append(sid)
    return speakers


def get_all_clips(speaker_dir: Path):
    clips = []
    for f in sorted(speaker_dir.rglob("*")):
        if f.suffix.lower() not in AUDIO_EXTENSIONS or not f.is_file():
            continue
        try:
            info = sf.info(str(f))
            duration = info.frames / info.samplerate
            clips.append((f, duration))
        except Exception as e:
            print(f"WARNING: failed to read {f}: {e}", file=sys.stderr)
    return clips


def select_toward_total(clips, min_total, min_count, max_count):
    """
    Shuffles clips and greedily adds them until the cumulative duration
    reaches min_total AND at least min_count clips are selected.
    Stops as soon as that's true. Never exceeds max_count clips.

    On the LAST available slot, if the target hasn't been met yet,
    prefers a clip that can close the remaining gap on its own
    (duration >= remaining_needed) instead of a purely random pick,
    to maximize the chance of hitting the target before running out
    of slots.

    Returns (selected_list, total_duration).
    """
    pool = clips[:]
    random.shuffle(pool)
    remaining_pool = pool[:]

    selected = []
    total = 0.0

    while remaining_pool:
        if len(selected) >= max_count:
            break

        is_last_slot = len(selected) == max_count - 1
        target_met = len(selected) >= min_count and total >= min_total

        if target_met:
            break

        if is_last_slot and total < min_total:
            # This is the final chance to hit the target — prefer a
            # clip whose own duration covers the remaining gap.
            needed = min_total - total
            covering = [c for c in remaining_pool if c[1] >= needed]

            if covering:
                # Prefer the smallest clip that still covers the gap,
                # so we don't overshoot more than necessary.
                covering.sort(key=lambda c: c[1])
                clip = covering[0]
            else:
                # Nothing can fully close the gap — take the largest
                # remaining clip to get as close as possible.
                remaining_pool.sort(key=lambda c: c[1], reverse=True)
                clip = remaining_pool[0]

            remaining_pool.remove(clip)
        else:
            clip = remaining_pool.pop(0)

        selected.append(clip)
        total += clip[1]

        if len(selected) >= min_count and total >= min_total:
            break

    return selected, total


def main():
    parser = argparse.ArgumentParser(
        description="Select and move reference clips per speaker"
    )
    parser.add_argument(
        "--csv", required=True, help="Path to elevenlabs_selected_speakers.csv"
    )
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Root folder containing speaker subfolders of bonafide audio",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root folder (e.g. elevenlabs-reference)",
    )
    parser.add_argument(
        "--min-total-duration",
        type=float,
        default=12.0,
        help=(
            "Minimum TOTAL duration (seconds) across selected "
            "clips per speaker (default 12s to buffer for "
            "leading-silence trimming)"
        ),
    )
    parser.add_argument(
        "--min-per-speaker",
        type=int,
        default=1,
        help=(
            "Minimum number of clips to select, even if total "
            "duration is already met"
        ),
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=5,
        help=(
            "Never select more than this many clips, even if "
            "total duration isn't met"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selections without moving files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    csv_path = Path(args.csv)
    audio_root = Path(args.audio_root)
    out_root = Path(args.out)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not audio_root.exists():
        print(f"ERROR: audio root not found: {audio_root}", file=sys.stderr)
        sys.exit(1)

    speakers = load_speaker_ids(csv_path)
    print(f"Speakers listed in CSV: {len(speakers)}")
    print("-" * 60)

    out_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for speaker_id in speakers:
        speaker_dir = audio_root / speaker_id

        if not speaker_dir.exists():
            print(
                f"[{speaker_id}] WARNING: no audio folder found at"
                f" {speaker_dir}",
                file=sys.stderr,
            )
            summary.append((speaker_id, 0, "missing folder"))
            continue

        clips = get_all_clips(speaker_dir)

        if not clips:
            print(
                f"[{speaker_id}] WARNING: no audio files found",
                file=sys.stderr,
            )
            summary.append((speaker_id, 0, 0.0, "no clips found"))
            continue

        selected, total = select_toward_total(
            clips,
            min_total=args.min_total_duration,
            min_count=args.min_per_speaker,
            max_count=args.max_per_speaker,
        )

        met_target = total >= args.min_total_duration
        note = (
            f"{len(clips)} clips available, total"
            f" {'reached' if met_target else 'FELL SHORT of'}"
            f" {args.min_total_duration}s target"
        )
        if not met_target:
            print(
                f"[{speaker_id}] WARNING: only reached {total:.2f}s across"
                f" {len(selected)} clips (target {args.min_total_duration}s,"
                f" max {args.max_per_speaker} clips)",
                file=sys.stderr,
            )

        dest_dir = out_root / speaker_id
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for src_path, duration in selected:
            dest_path = dest_dir / src_path.name
            if args.dry_run:
                print(
                    f"[{speaker_id}] WOULD MOVE  {src_path}  ({duration:.2f}s)"
                    f" -> {dest_path}"
                )
            else:
                shutil.move(str(src_path), str(dest_path))
                print(
                    f"[{speaker_id}] moved {src_path.name} ({duration:.2f}s)"
                )

        print(
            f"[{speaker_id}] selected {len(selected)} clips, total"
            f" {total:.2f}s"
        )
        summary.append((speaker_id, len(selected), total, note))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for speaker_id, n_selected, total_dur, note in summary:
        print(
            f"  [{speaker_id}] selected {n_selected} clips, {total_dur:.2f}s"
            f" total  ({note})"
        )

    total_selected = sum(n for _, n, _, _ in summary)
    print(
        f"\nTotal clips {'that would be' if args.dry_run else ''} moved:"
        f" {total_selected}"
    )
    print(f"Destination: {out_root.resolve()}")

    if args.dry_run:
        print("\n(dry run — nothing was actually moved)")


if __name__ == "__main__":
    main()
