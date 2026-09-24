r"""
make_loso_splits.py

Derives leave-one-system-out manifests from the existing splits, to
test whether the detector generalises to a synthesis system it has
never seen.

WHY THIS EXISTS

build_manifests.py makes the split speaker-disjoint, and it is -- on
the bonafide side. Each bonafide speaker is tied to one synthesis
method upstream, and their spoof clips travel with them, so no
bonafide speaker appears in two splits.

But the Meta MMS voice is not a bonafide speaker. It is a single
synthetic voice wearing whatever speaker_id the utterance it cloned
happened to carry. So that one voice appears in train, val AND test,
and speaker-disjointness cannot prevent it, because the repeated thing
is not in the speaker column.

A detector can therefore learn "this timbre is spoof" and score well
on roughly half the spoof class without learning anything about
synthesis. That is speaker identification wearing a detector's coat,
and it generalises to nothing.

probe_shortcuts.py cannot see this. It has no voice features at all.
The only way to measure it is to hold a system out of training
entirely and see what survives.

WHAT IT BUILDS

For each spoof system S, a directory containing four manifests:

    train.parquet          bonafide + every system EXCEPT S
    val.parquet            same composition -- checkpoint selection
                           must happen on SEEN systems, or you are
                           selecting on your own test condition
    test_matched.parquet   bonafide + every system EXCEPT S
    test_unseen.parquet    bonafide + S only

The number that matters is the GAP between test_matched and
test_unseen for the same checkpoint:

    small gap    the model learned synthesis artefacts. This is the
                 result that lets you claim a detector rather than a
                 voice classifier.

    large gap    the model memorised the systems it trained on. Report
                 it -- with two systems and one MMS voice this is a
                 real possibility, and a measured limitation is worth
                 far more than an unexamined headline number.

This mirrors standard practice: ASVspoof's evaluation attacks are
deliberately disjoint from its training attacks for exactly this
reason.

BALANCE

Dropping a system halves the spoof side, so bonafide is subsampled to
match, stratified by speaker so no speaker disappears. The LOSO runs
therefore see less data than the main run and their absolute EERs are
NOT comparable to it. Compare test_matched against test_unseen within
a run; that contrast is internally valid because both use the same
checkpoint and the same bonafide pool.

Usage:

    python training\make_loso_splits.py --splits-dir manifests\splits_norm --out manifests\loso

    # keep every bonafide clip instead of balancing
    python training\make_loso_splits.py --splits-dir manifests\splits_norm --out manifests\loso --no-balance
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------


def load_splits(splits_dir):
    """Read the three parquets and normalise the label column to 0/1."""
    frames = {}

    for name in SPLITS:
        path = Path(splits_dir) / f"{name}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build the base manifests first."
            )

        df = pd.read_parquet(path)

        missing = {"path", "label", "speaker_id", "spoof_method"} - set(
            df.columns
        )

        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        if pd.api.types.is_numeric_dtype(df["label"]):
            df["is_bonafide"] = df["label"].astype(int) == 1
        else:
            values = df["label"].astype(str).str.strip().str.lower()
            unknown = set(values.unique()) - {"bonafide", "spoof"}

            if unknown:
                raise ValueError(f"unexpected labels: {sorted(unknown)}")

            df["is_bonafide"] = values == "bonafide"

        frames[name] = df

    return frames


def spoof_systems(frames):
    """Every distinct spoof_method present across the splits."""
    found = set()

    for df in frames.values():
        spoof = df.loc[~df["is_bonafide"], "spoof_method"].dropna()
        found.update(spoof.unique())

    return sorted(found)


# ---------------------------------------------------------------------------
# COMPOSITION
# ---------------------------------------------------------------------------


def balance_bonafide(df, n_target, seed):
    """
    Subsample bonafide rows to n_target, proportionally per speaker.

    Per-speaker rather than a flat random draw so that no speaker is
    dropped entirely -- losing speakers would change the bonafide
    distribution between the main run and the LOSO runs for a reason
    that has nothing to do with the held-out system.
    """
    bona = df[df["is_bonafide"]]

    if n_target >= len(bona):
        return bona

    frac = n_target / len(bona)

    # A plain loop rather than groupby.apply: the apply signature has
    # changed across pandas versions (include_groups was deprecated and
    # then removed), and this script has to run on whatever pandas the
    # training environment happens to have.
    parts = [
        group.sample(n=max(1, round(len(group) * frac)), random_state=seed)
        for _, group in bona.groupby("speaker_id", sort=True)
    ]

    picked = pd.concat(parts)

    # Rounding per speaker will not land exactly on the target.
    if len(picked) > n_target:
        picked = picked.sample(n=n_target, random_state=seed)
    elif len(picked) < n_target:
        spare = bona.drop(index=picked.index)
        need = min(n_target - len(picked), len(spare))

        if need > 0:
            picked = pd.concat(
                [picked, spare.sample(n=need, random_state=seed)]
            )

    return picked


def compose(df, keep_systems, balance, seed):
    """bonafide + spoofs from keep_systems, optionally class-balanced."""
    spoof = df[~df["is_bonafide"] & df["spoof_method"].isin(keep_systems)]

    if balance:
        bona = balance_bonafide(df, len(spoof), seed)
    else:
        bona = df[df["is_bonafide"]]

    out = pd.concat([bona, spoof])

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------


def check_disjoint(built, held_out):
    """
    Confirm the derived manifests kept the guarantees that matter.

    Filtering inherits speaker-disjointness from the source splits, but
    inheriting a property is not the same as having verified it, and a
    silent violation here would invalidate the whole run.
    """
    problems = []

    speakers = {
        name: set(df["speaker_id"].dropna()) for name, df in built.items()
    }

    for a, b in (
        ("train", "val"),
        ("train", "test_matched"),
        ("train", "test_unseen"),
        ("val", "test_matched"),
        ("val", "test_unseen"),
    ):
        overlap = speakers[a] & speakers[b]

        if overlap:
            problems.append(
                f"{a} and {b} share {len(overlap)} speaker(s): "
                f"{sorted(overlap)[:5]}"
            )

    # The whole point: the held-out system must not be in training.
    for name in ("train", "val", "test_matched"):
        leaked = built[name]["spoof_method"].eq(held_out).sum()

        if leaked:
            problems.append(f"{name} contains {leaked} {held_out} clips")

    unseen_systems = set(
        built["test_unseen"]["spoof_method"].dropna().unique()
    )

    if unseen_systems != {held_out}:
        problems.append(
            f"test_unseen should contain only {held_out}, "
            f"has {sorted(unseen_systems)}"
        )

    for name, df in built.items():
        if df["is_bonafide"].all():
            problems.append(f"{name} has no spoof clips")
        elif not df["is_bonafide"].any():
            problems.append(f"{name} has no bonafide clips")

    # test_matched and test_unseen SHOULD overlap -- they share one
    # bonafide pool on purpose. What must hold is that they share it
    # exactly, and carry the same number of spoofs, so the gap between
    # their EERs is attributable only to the held-out system.
    matched, unseen = built["test_matched"], built["test_unseen"]

    bona_m = set(matched.loc[matched["is_bonafide"], "path"])
    bona_u = set(unseen.loc[unseen["is_bonafide"], "path"])

    if bona_m != bona_u:
        problems.append(
            "test_matched and test_unseen use different bonafide clips "
            f"({len(bona_m ^ bona_u)} differ) -- the gap would be "
            "confounded"
        )

    n_m = int((~matched["is_bonafide"]).sum())
    n_u = int((~unseen["is_bonafide"]).sum())

    if n_m != n_u:
        problems.append(
            f"test_matched has {n_m} spoofs, test_unseen {n_u} -- "
            "equalise or the gap is partly a size effect"
        )

    spoof_m = set(matched.loc[~matched["is_bonafide"], "path"])
    spoof_u = set(unseen.loc[~unseen["is_bonafide"], "path"])

    if spoof_m & spoof_u:
        problems.append("test_matched and test_unseen share spoof clips")

    return problems


def describe(name, df):
    bona = int(df["is_bonafide"].sum())
    spoof = int((~df["is_bonafide"]).sum())
    systems = sorted(
        df.loc[~df["is_bonafide"], "spoof_method"].dropna().unique()
    )
    speakers = df["speaker_id"].nunique()

    return (
        f"    {name:<16} {len(df):>7} clips  "
        f"bonafide={bona:<6} spoof={spoof:<6} "
        f"speakers={speakers:<4} {','.join(systems) or '-'}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def build_test_pair(test_df, held_out, others, balance, seed):
    """
    Build test_matched and test_unseen as a matched pair.

    The whole experiment is the DIFFERENCE between these two numbers,
    so the bonafide side has to be identical in both and the spoof
    counts equal. Otherwise part of the gap is just the two sets having
    different composition, and you cannot attribute it to the held-out
    system.

    Both spoof sides are therefore trimmed to the smaller of the two,
    and one bonafide sample is drawn and reused.
    """
    spoof_matched = test_df[
        ~test_df["is_bonafide"] & test_df["spoof_method"].isin(others)
    ]
    spoof_unseen = test_df[
        ~test_df["is_bonafide"] & test_df["spoof_method"].eq(held_out)
    ]

    if balance:
        n_spoof = min(len(spoof_matched), len(spoof_unseen))
        spoof_matched = spoof_matched.sample(n=n_spoof, random_state=seed)
        spoof_unseen = spoof_unseen.sample(n=n_spoof, random_state=seed)
        bona = balance_bonafide(test_df, n_spoof, seed)
    else:
        bona = test_df[test_df["is_bonafide"]]

    def finish(spoof):
        out = pd.concat([bona, spoof])
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return finish(spoof_matched), finish(spoof_unseen)


def build_one(frames, held_out, others, args):
    """Build the four manifests for one held-out system."""
    matched, unseen = build_test_pair(
        frames["test"], held_out, others, args.balance, args.seed
    )

    return {
        "train": compose(frames["train"], others, args.balance, args.seed),
        "val": compose(frames["val"], others, args.balance, args.seed),
        "test_matched": matched,
        "test_unseen": unseen,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Derive leave-one-system-out manifests, to measure whether "
            "the detector generalises to an unseen synthesis system."
        )
    )

    parser.add_argument(
        "--splits-dir",
        default="manifests/splits_norm",
        help="directory holding train/val/test.parquet",
    )
    parser.add_argument("--out", default="manifests/loso")
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help=(
            "keep every bonafide clip. Default subsamples bonafide to "
            "match the reduced spoof count."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    try:
        frames = load_splits(args.splits_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    systems = spoof_systems(frames)

    print("=" * 78)
    print("LEAVE-ONE-SYSTEM-OUT SPLITS")
    print("=" * 78)
    print()
    print(f"  source     : {args.splits_dir}")
    print(f"  systems    : {', '.join(systems)}")
    print(f"  balance    : {'yes' if args.balance else 'no'}")
    print()

    if len(systems) < 2:
        print(
            f"ERROR: need at least 2 spoof systems, found {systems}. "
            "With one system there is nothing to hold out.",
            file=sys.stderr,
        )
        return 1

    failed = False

    for held_out in systems:
        others = [s for s in systems if s != held_out]
        built = build_one(frames, held_out, others, args)

        print("-" * 78)
        print(f"  HOLD OUT: {held_out}     (train on: {', '.join(others)})")
        print("-" * 78)

        for name in ("train", "val", "test_matched", "test_unseen"):
            print(describe(name, built[name]))

        problems = check_disjoint(built, held_out)

        if problems:
            failed = True
            print()
            for p in problems:
                print(f"    FAIL: {p}", file=sys.stderr)
        else:
            print(
                "    checks: speaker-disjoint, held-out system absent "
                "from train/val/matched"
            )

        out_dir = Path(args.out) / f"holdout_{held_out}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, df in built.items():
            df.drop(columns=["is_bonafide"]).to_parquet(
                out_dir / f"{name}.parquet", index=False
            )

        print(f"    written: {out_dir}")
        print()

    if failed:
        print("RESULT: FAILED -- do not train on these.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("NEXT")
    print("=" * 78)
    print()
    print("  One training run per held-out system. Copy train_main.yaml")
    print("  and change ONLY the parquet paths -- every other value must")
    print("  match, or the comparison measures your config, not the")
    print("  held-out system.")
    print()

    for held_out in systems:
        d = (Path(args.out) / f"holdout_{held_out}").as_posix()
        print(f"  holdout_{held_out}:")
        print(f'    train: parquet_files: ["{d}/train.parquet"]')
        print(f'    val:   parquet_files: ["{d}/val.parquet"]')
        print()

    print("  Then score EACH checkpoint against BOTH test sets. The")
    print("  headline of this experiment is the gap:")
    print()
    print("      EER(test_unseen) - EER(test_matched)")
    print()
    print("  A small gap means synthesis artefacts. A large gap means")
    print("  the model memorised the systems it saw -- which, with one")
    print("  MMS voice, is the outcome to watch for.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
