r"""
audit_sources.py

Audits which transcript sources are actually present in the corpus and,
optionally, quarantines unwanted ones.

Runs in AUDIT MODE by default. Nothing is moved, deleted, or modified
unless --apply is passed explicitly.

Why this exists:

    EXCLUDE_SOURCE_PREFIXES ("TGL_", "CEB_Utt_Eng") is already applied
    by inventory_bonafide.py, resample_dataset.py, and
    extract_transcripts.py. This script verifies that the exclusions
    actually landed, and surfaces what is still present.

    It also catches the silent gap in those scripts: a clip whose
    filename has no matching entry in its speaker's .log gets
    source_file = None and is INCLUDED by default. Those clips are
    reported here as "<no source in log>".

Audit output:

    Every distinct transcript source in the corpus, with clip count,
    speaker count, and total duration -- so you can see exactly what
    the dataset is made of before deciding to cut anything.

    A cross-check against a processed audio root (--processed-root)
    reports whether any already-excluded source survived into the
    processed set.

Quarantine (--apply):

    Matching clips are MOVED into an _excluded/ folder beside the
    processed root, preserving speaker subfolders. Files are never
    deleted, so any cut is reversible by moving them back.

    IMPORTANT -- class symmetry:

        Bonafide transcripts also drive ElevenLabs and Meta MMS
        generation. Removing a source from bonafide ONLY would leave
        spoof clips whose bonafide counterparts are gone, and a
        detector can exploit that asymmetry as a shortcut.

        Pass --spoof-root for each spoof set so the matching
        .1.wav (Meta MMS) and .2.wav (ElevenLabs) clips are
        quarantined alongside their bonafide source.

Filename convention:

    bonafide    0201.111024.022631.0430.wav
    Meta MMS    0201.111024.022631.0430.1.wav
    ElevenLabs  0201.111024.022631.0430.2.wav

Requires:

    pip install soundfile

Usage:

    # Audit only -- nothing is touched
    python preprocessing\audit_sources.py \
        --root "C:\path\to\original\bonafide"

    # Audit, and cross-check what survived into the processed set
    python preprocessing\audit_sources.py \
        --root "C:\path\to\original\bonafide" \
        --processed-root data\processed\bonafide \
        --out manifests\source_audit.csv

    # Preview a cut (still does not move anything)
    python preprocessing\audit_sources.py \
        --root "C:\path\to\original\bonafide" \
        --processed-root data\processed\bonafide \
        --exclude-prefix CEB_Iso_

    # Actually quarantine, keeping spoof sets in sync
    python preprocessing\audit_sources.py \
        --root "C:\path\to\original\bonafide" \
        --processed-root data\processed\bonafide \
        --spoof-root data\processed\meta-mms \
        --spoof-root data\processed\elevenlabs \
        --exclude-prefix CEB_Iso_ \
        --apply
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print(
        "Missing dependency. Run: pip install soundfile",
        file=sys.stderr,
    )
    sys.exit(1)


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Mirrors the tuple already used by inventory_bonafide.py,
# resample_dataset.py, and extract_transcripts.py. Reported as
# "already excluded" so you can confirm those filters actually landed.
KNOWN_EXCLUDED_PREFIXES = (
    "TGL_",
    "CEB_Utt_Eng",
)

NO_SOURCE_LABEL = "<no source in log>"

QUARANTINE_DIRNAME = "_excluded"


# ---------------------------------------------------------------------------
# LOG PARSING
# ---------------------------------------------------------------------------


def find_log_file(speaker_dir: Path):
    """Find the .log file directly inside a speaker folder."""
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


def parse_transcript_sources(log_path: Path) -> dict:
    """
    Returns:

        {
            "clip.wav": "TGL_spontaneous.txt",
            "other.wav": "CEB_Iso_Cities.txt",
        }

    Same parsing shape as resample_dataset.py, kept local so this
    script stays runnable on its own.
    """
    sources = {}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
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

                sources[filename] = parts[1].strip()

    except Exception as e:
        print(
            f"WARNING: failed to parse {log_path}: {e}",
            file=sys.stderr,
        )

    return sources


def is_known_excluded(source_file: str) -> bool:
    return any(
        source_file.startswith(prefix) for prefix in KNOWN_EXCLUDED_PREFIXES
    )


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------


def get_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def audit_corpus(root: Path, measure_duration: bool) -> dict:
    """
    Walks every speaker folder and tallies clips per transcript source.

    Returns:

        {
            source_file: {
                "clips": int,
                "speakers": set[str],
                "duration": float,
                "files": [(speaker_id, filename)],
            }
        }
    """
    stats = defaultdict(
        lambda: {
            "clips": 0,
            "speakers": set(),
            "duration": 0.0,
            "files": [],
        }
    )

    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not speaker_dirs:
        print(
            f"ERROR: no speaker subfolders under {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning {len(speaker_dirs)} speaker folders...")

    for index, speaker_dir in enumerate(speaker_dirs, start=1):
        speaker_id = speaker_dir.name

        log_file = find_log_file(speaker_dir)

        if log_file is None:
            print(
                f"WARNING: [{speaker_id}] no .log file found",
                file=sys.stderr,
            )
            transcript_sources = {}
        else:
            transcript_sources = parse_transcript_sources(log_file)

        for path in sorted(speaker_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            source_file = transcript_sources.get(path.name, NO_SOURCE_LABEL)

            entry = stats[source_file]
            entry["clips"] += 1
            entry["speakers"].add(speaker_id)
            entry["files"].append((speaker_id, path.name))

            if measure_duration:
                entry["duration"] += get_duration(path)

        if index % 20 == 0 or index == len(speaker_dirs):
            print(f"  {index}/{len(speaker_dirs)} speakers scanned")

    return stats


def print_audit(stats: dict, measure_duration: bool):
    total_clips = sum(e["clips"] for e in stats.values())

    print()
    print("=" * 70)
    print("TRANSCRIPT SOURCE AUDIT")
    print("=" * 70)
    print(f"Distinct sources : {len(stats)}")
    print(f"Total clips      : {total_clips}")
    print()

    header = f"{'source':42s} {'clips':>7s} {'spk':>5s} {'%':>6s}"

    if measure_duration:
        header += f" {'hours':>7s}"

    print(header)
    print("-" * 70)

    for source, entry in sorted(
        stats.items(), key=lambda kv: kv[1]["clips"], reverse=True
    ):
        pct = 100 * entry["clips"] / total_clips if total_clips else 0

        flag = ""
        if source == NO_SOURCE_LABEL:
            flag = "  <-- kept by default, NOT filtered"
        elif is_known_excluded(source):
            flag = "  <-- already excluded by pipeline"

        line = (
            f"{source[:42]:42s} "
            f"{entry['clips']:7d} "
            f"{len(entry['speakers']):5d} "
            f"{pct:5.1f}%"
        )

        if measure_duration:
            line += f" {entry['duration'] / 3600:7.2f}"

        print(line + flag)

    print("-" * 70)


def cross_check_processed(stats: dict, processed_root: Path):
    """
    Reports whether clips from already-excluded sources survived into
    the processed audio set. This is the empirical answer to "are the
    exclusions actually working?"
    """
    print()
    print("=" * 70)
    print("CROSS-CHECK AGAINST PROCESSED SET")
    print("=" * 70)
    print(f"Processed root: {processed_root}")

    if not processed_root.exists():
        print(
            f"WARNING: processed root not found: {processed_root}",
            file=sys.stderr,
        )
        return

    present = {
        path.name
        for path in processed_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }

    print(f"Files in processed set: {len(present)}")
    print()

    leaked_total = 0

    for source, entry in sorted(stats.items()):
        if not is_known_excluded(source):
            continue

        leaked = [
            filename for _, filename in entry["files"] if filename in present
        ]

        status = "CLEAN" if not leaked else f"{len(leaked)} LEAKED"

        print(f"  {source[:44]:44s} {entry['clips']:6d} in corpus -> {status}")

        leaked_total += len(leaked)

        for filename in leaked[:5]:
            print(f"      still present: {filename}")

    no_source = stats.get(NO_SOURCE_LABEL)

    if no_source:
        kept = [
            filename
            for _, filename in no_source["files"]
            if filename in present
        ]
        print()
        print(
            "  Clips with no .log source that reached the "
            f"processed set: {len(kept)}"
        )
        print(
            "      These were never source-filtered -- review them "
            "before training."
        )

    print()

    if leaked_total == 0:
        print("RESULT: no excluded-source clips found in the processed set.")
    else:
        print(
            f"RESULT: {leaked_total} clip(s) from excluded sources "
            "are still present."
        )


# ---------------------------------------------------------------------------
# QUARANTINE
# ---------------------------------------------------------------------------


def spoof_variants(bonafide_filename: str) -> list[str]:
    """
    0201.111024.022631.0430.wav
        -> ["0201.111024.022631.0430.1.wav",
            "0201.111024.022631.0430.2.wav"]
    """
    stem = Path(bonafide_filename).stem
    return [f"{stem}.1.wav", f"{stem}.2.wav"]


def collect_targets(
    stats: dict,
    exclude_prefixes: tuple,
    exclude_sources: tuple,
) -> list:
    """Returns [(speaker_id, filename, source_file)] to quarantine."""
    targets = []

    for source, entry in stats.items():
        hit_prefix = any(source.startswith(p) for p in exclude_prefixes)
        hit_exact = source in exclude_sources

        if hit_prefix or hit_exact:
            for speaker_id, filename in entry["files"]:
                targets.append((speaker_id, filename, source))

    return targets


def quarantine(
    targets: list,
    roots: list,
    apply_changes: bool,
) -> dict:
    """
    Moves each target (and its spoof counterparts) into
    <root>/../_excluded/<speaker_id>/.

    With apply_changes=False nothing is moved -- the planned actions
    are printed instead.
    """
    counts = defaultdict(int)

    for root in roots:
        if not root.exists():
            print(
                f"WARNING: root not found, skipping: {root}",
                file=sys.stderr,
            )
            continue

        quarantine_root = root.parent / QUARANTINE_DIRNAME / root.name

        for speaker_id, filename, source in targets:
            candidates = [filename] + spoof_variants(filename)

            for candidate in candidates:
                src = root / speaker_id / candidate

                if not src.exists():
                    continue

                dst = quarantine_root / speaker_id / candidate

                counts[str(root)] += 1

                if not apply_changes:
                    print(f"  WOULD MOVE  {src}  ->  {dst}")
                    continue

                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                print(f"  MOVED  {src}  ->  {dst}")

    return counts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit transcript sources in the corpus and optionally "
            "quarantine unwanted ones. Audit-only by default."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Original corpus root (speaker folders with .log files)",
    )

    parser.add_argument(
        "--processed-root",
        default=None,
        help=(
            "Processed bonafide audio root. If given, cross-checks "
            "whether excluded sources survived into it."
        ),
    )

    parser.add_argument(
        "--spoof-root",
        action="append",
        default=[],
        help=(
            "Spoof audio root to keep in sync when quarantining. "
            "Repeatable, e.g. --spoof-root data\\processed\\meta-mms "
            "--spoof-root data\\processed\\elevenlabs"
        ),
    )

    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help=(
            "Quarantine sources starting with this prefix. "
            "Repeatable. Example: --exclude-prefix CEB_Iso_"
        ),
    )

    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help=(
            "Quarantine one exact source filename. Repeatable. "
            "Example: --exclude-source CEB_Iso_Cities.txt"
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV path for the per-source audit table",
    )

    parser.add_argument(
        "--durations",
        action="store_true",
        help=(
            "Measure total duration per source by opening every file "
            "(slower; adds an hours column)"
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually move the matched clips into _excluded/. "
            "Without this flag the script only reports what it would "
            "do and changes nothing."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    stats = audit_corpus(root, args.durations)

    print_audit(stats, args.durations)

    if args.processed_root:
        cross_check_processed(stats, Path(args.processed_root))

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source_file",
                    "clip_count",
                    "speaker_count",
                    "duration_sec",
                    "already_excluded_by_pipeline",
                ]
            )

            for source, entry in sorted(
                stats.items(),
                key=lambda kv: kv[1]["clips"],
                reverse=True,
            ):
                writer.writerow(
                    [
                        source,
                        entry["clips"],
                        len(entry["speakers"]),
                        f"{entry['duration']:.3f}",
                        is_known_excluded(source),
                    ]
                )

        print()
        print(f"Audit table written to: {out_path.resolve()}")

    # -----------------------------------------------------------------
    # QUARANTINE
    # -----------------------------------------------------------------

    exclude_prefixes = tuple(args.exclude_prefix)
    exclude_sources = tuple(args.exclude_source)

    if not exclude_prefixes and not exclude_sources:
        print()
        print(
            "Audit only -- no --exclude-prefix / --exclude-source "
            "given, nothing to quarantine."
        )
        return

    targets = collect_targets(stats, exclude_prefixes, exclude_sources)

    print()
    print("=" * 70)
    print("QUARANTINE PLAN" if not args.apply else "QUARANTINE")
    print("=" * 70)
    print(f"Prefixes : {exclude_prefixes or '(none)'}")
    print(f"Sources  : {exclude_sources or '(none)'}")
    print(f"Matched bonafide clips: {len(targets)}")

    if not targets:
        print("Nothing matched. No action taken.")
        return

    roots = []

    if args.processed_root:
        roots.append(Path(args.processed_root))

    roots.extend(Path(p) for p in args.spoof_root)

    if not roots:
        print(
            "\nERROR: --processed-root and/or --spoof-root required "
            "to quarantine.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.apply:
        print(
            "\nDRY RUN -- nothing will be moved. "
            "Re-run with --apply to execute.\n"
        )

    counts = quarantine(targets, roots, args.apply)

    print()
    print("-" * 70)

    for root_name, n in sorted(counts.items()):
        verb = "moved" if args.apply else "would move"
        print(f"  {root_name}: {n} file(s) {verb}")

    total = sum(counts.values())

    if args.apply:
        print(f"\nDone. {total} file(s) moved into {QUARANTINE_DIRNAME}/.")
        print("Files were MOVED, not deleted -- move them back to undo.")
    else:
        print(f"\nDRY RUN complete. {total} file(s) would be moved.")
        print("Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
