"""
select_mms_speakers.py

Selects a subset of speakers (default: half) from the bonafide manifest,
stratified by gender so the selected subset preserves the same male/female
ratio as the full speaker pool. Useful for deciding which speakers'
utterances get sent through Meta MMS for spoof generation.

Uses a fixed random seed for reproducibility -- rerunning with the same
seed and manifest always gives the same selection.

Usage:
    python preprocessing\select_speakers.py --manifest manifests\\manifest_bonafide.csv \
        --fraction 0.5 --seed 42 --out manifests\\mms_selected_speakers.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict


def load_speaker_gender(manifest_path: Path) -> dict:
    """Returns {speaker_id: gender} using one row per unique speaker."""
    speaker_gender = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "speaker_id" not in reader.fieldnames or "speaker_gender" not in reader.fieldnames:
            print("ERROR: manifest must have 'speaker_id' and 'speaker_gender' columns.",
                  file=sys.stderr)
            sys.exit(1)
        for row in reader:
            spk = row["speaker_id"]
            gender = row.get("speaker_gender", "").strip()
            if spk not in speaker_gender and gender:
                speaker_gender[spk] = gender
    return speaker_gender


def load_clip_stats(manifest_path: Path) -> dict:
    """
    Returns {speaker_id: {"clip_count": int, "duration_sec": float}}
    by scanning every row (not just unique speakers) in the manifest.
    """
    stats = defaultdict(lambda: {"clip_count": 0, "duration_sec": 0.0})
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spk = row["speaker_id"]
            stats[spk]["clip_count"] += 1
            try:
                stats[spk]["duration_sec"] += float(row.get("duration_sec", 0) or 0)
            except ValueError:
                pass
    return stats


def stratified_select(speaker_gender: dict, fraction: float, seed: int) -> dict:
    """
    Selects `fraction` of speakers within each gender group, preserving ratio.
    Returns {speaker_id: gender} for the selected subset.
    """
    by_gender = defaultdict(list)
    for spk, gender in speaker_gender.items():
        by_gender[gender].append(spk)

    rng = random.Random(seed)
    selected = {}

    for gender, speakers in by_gender.items():
        speakers_sorted = sorted(speakers)  # sort first for deterministic shuffling
        rng.shuffle(speakers_sorted)
        n_select = round(len(speakers_sorted) * fraction)
        chosen = speakers_sorted[:n_select]
        for spk in chosen:
            selected[spk] = gender

    return selected


def write_selection(selected: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_id", "speaker_gender"])
        for spk, gender in sorted(selected.items()):
            writer.writerow([spk, gender])


def write_complement(speaker_gender: dict, selected: dict, out_path: Path):
    """Writes the speakers NOT selected -- i.e. the other engine's group."""
    complement = {spk: gender for spk, gender in speaker_gender.items() if spk not in selected}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_id", "speaker_gender"])
        for spk, gender in sorted(complement.items()):
            writer.writerow([spk, gender])
    return complement


def print_summary(speaker_gender: dict, selected: dict, fraction: float, seed: int,
                   clip_stats: dict = None):
    from collections import Counter
    total_counts = Counter(speaker_gender.values())
    selected_counts = Counter(selected.values())

    print("=" * 60)
    print(f"Stratified speaker selection (fraction={fraction}, seed={seed})")
    print("=" * 60)
    print(f"Total speakers        : {len(speaker_gender)}")
    print(f"  {dict(total_counts)}")
    print(f"Selected speakers     : {len(selected)}")
    print(f"  {dict(selected_counts)}")

    print("\nSpeaker-level ratio check:")
    for gender in total_counts:
        total_n = total_counts[gender]
        sel_n = selected_counts.get(gender, 0)
        orig_pct = 100 * total_n / len(speaker_gender)
        sel_pct = 100 * sel_n / len(selected) if selected else 0
        print(f"  {gender}: {orig_pct:.1f}% of full pool -> {sel_pct:.1f}% of selected "
              f"({sel_n}/{total_n} speakers selected)")

    if clip_stats:
        # This is the part that actually matters for your MMS budget --
        # speaker-level balance doesn't guarantee clip-level balance,
        # since speakers don't all have the same number of clips.
        clip_counts_by_gender = Counter()
        duration_by_gender = Counter()
        for spk, gender in selected.items():
            clip_counts_by_gender[gender] += clip_stats.get(spk, {}).get("clip_count", 0)
            duration_by_gender[gender] += clip_stats.get(spk, {}).get("duration_sec", 0.0)

        total_selected_clips = sum(clip_counts_by_gender.values())
        total_selected_duration_hr = sum(duration_by_gender.values()) / 3600

        print("\nResulting CLIP-level ratio -- SELECTED group (engine A, e.g. MMS):")
        for gender in clip_counts_by_gender:
            n = clip_counts_by_gender[gender]
            pct = 100 * n / total_selected_clips if total_selected_clips else 0
            hrs = duration_by_gender[gender] / 3600
            print(f"  {gender}: {n} clips ({pct:.1f}%), {hrs:.2f} hours")
        print(f"  TOTAL: {total_selected_clips} clips, {total_selected_duration_hr:.2f} hours")

        # flag if clip-level ratio has drifted meaningfully from speaker-level ratio
        for gender in total_counts:
            speaker_pct = 100 * selected_counts.get(gender, 0) / len(selected) if selected else 0
            clip_pct = 100 * clip_counts_by_gender.get(gender, 0) / total_selected_clips \
                if total_selected_clips else 0
            drift = abs(speaker_pct - clip_pct)
            if drift > 5:
                print(f"\n⚠ {gender}: speaker ratio ({speaker_pct:.1f}%) and clip ratio "
                      f"({clip_pct:.1f}%) differ by {drift:.1f} points -- some speakers "
                      f"in this group likely have disproportionately more/fewer clips.")


def print_complement_summary(speaker_gender: dict, complement: dict, clip_stats: dict = None):
    """Same breakdown as print_summary, but for the unselected (engine B) group."""
    from collections import Counter
    complement_counts = Counter(complement.values())

    print("\n" + "=" * 60)
    print("Complement group (engine B, e.g. ElevenLabs)")
    print("=" * 60)
    print(f"Complement speakers    : {len(complement)}")
    print(f"  {dict(complement_counts)}")

    if clip_stats:
        clip_counts_by_gender = Counter()
        duration_by_gender = Counter()
        for spk, gender in complement.items():
            clip_counts_by_gender[gender] += clip_stats.get(spk, {}).get("clip_count", 0)
            duration_by_gender[gender] += clip_stats.get(spk, {}).get("duration_sec", 0.0)

        total_clips = sum(clip_counts_by_gender.values())
        total_duration_hr = sum(duration_by_gender.values()) / 3600

        print("\nResulting CLIP-level ratio -- COMPLEMENT group:")
        for gender in clip_counts_by_gender:
            n = clip_counts_by_gender[gender]
            pct = 100 * n / total_clips if total_clips else 0
            hrs = duration_by_gender[gender] / 3600
            print(f"  {gender}: {n} clips ({pct:.1f}%), {hrs:.2f} hours")
        print(f"  TOTAL: {total_clips} clips, {total_duration_hr:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description="Stratified speaker selection for MMS processing")
    parser.add_argument("--manifest", required=True, help="Path to manifest_bonafide.csv")
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of speakers to select per gender (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", required=True, help="Output CSV of selected speakers")
    parser.add_argument("--out-complement", default=None,
                         help="Optional: also write the unselected speakers "
                              "(the other engine's group) to this CSV path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    speaker_gender = load_speaker_gender(manifest_path)
    if not speaker_gender:
        print("ERROR: no speakers with gender info found in manifest.", file=sys.stderr)
        sys.exit(1)

    selected = stratified_select(speaker_gender, args.fraction, args.seed)
    clip_stats = load_clip_stats(manifest_path)
    write_selection(selected, Path(args.out))
    print_summary(speaker_gender, selected, args.fraction, args.seed, clip_stats)
    print(f"\nSelected speaker list written to: {Path(args.out).resolve()}")

    if args.out_complement:
        complement = write_complement(speaker_gender, selected, Path(args.out_complement))
        print_complement_summary(speaker_gender, complement, clip_stats)
        print(f"\nComplement speaker list written to: {Path(args.out_complement).resolve()}")


if __name__ == "__main__":
    main()