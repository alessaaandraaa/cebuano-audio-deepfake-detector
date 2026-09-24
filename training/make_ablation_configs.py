r"""
make_ablation_configs.py

Generates the experiment-matrix configs from train_main.yaml, changing
one thing each and proving that nothing else moved.

WHY GENERATE RATHER THAN COPY

An ablation is only valid if the runs differ in exactly the variable
under test. Hand-copying five YAML files and editing each one is how a
stray batch_size or a forgotten seed ends up in the results table, and
you cannot tell from the numbers that it happened -- you get a clean
looking comparison that measures your config instead of your variable.

So this derives every config from one base, applies a named patch, and
then DIFFS the result back against the base. If anything changed beyond
the declared keys, it refuses to write the file.

THE MATRIX

    noaug              no codec augmentation at all
                       -> what does codec augmentation buy?

    opus               Opus only
    amr                AMR-NB only
                       -> does training on one codec transfer to the
                          other? (cross-codec generalisation)

    loso_<system>      trains with one synthesis system held out
                       entirely, using the manifests from
                       make_loso_splits.py
                       -> does the detector generalise to a system it
                          has never heard?

Cross-CONDITION evaluation does not need its own training run: score
one checkpoint against clean / Opus / AMR versions of the test set.
Only cross-CODEC-TRAINING and leave-one-system-out need separate runs.

COST

Each run is roughly 2h10m per epoch on the reference laptop (RTX 5060,
batch 4, ~79k clips). With early_stopping_patience 7 expect 10-20
epochs, so 24-48 h per run. Budget days, not hours, and read the
disk warning in the README before queueing several.

Usage:

    # see what would be written, write nothing
    python training\make_ablation_configs.py

    # write them
    python training\make_ablation_configs.py --apply

    # only some
    python training\make_ablation_configs.py --apply --only noaug opus
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------
#
# Each entry declares the dotted key paths it is allowed to touch. The
# verifier below fails the build if a patch changes anything else, so a
# typo in a patch function cannot silently widen the difference.

OPUS_BLOCK = {
    "type": "opus_codec",
    "noise_ratio": 0.5,
    "bitrate_min": 6000,
    "bitrate_max": 32000,
}

AMR_BLOCK = {
    "type": "amrnb_codec",
    "noise_ratio": 0.5,
    "bitrates": [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200],
}


def patch_noaug(cfg):
    cfg["exp_name"] = "ceb_noaug"
    cfg["data"]["train"].pop("augment_transform", None)
    return cfg


def patch_opus(cfg):
    cfg["exp_name"] = "ceb_opus_only"
    cfg["data"]["train"]["augment_transform"] = [copy.deepcopy(OPUS_BLOCK)]
    return cfg


def patch_amr(cfg):
    cfg["exp_name"] = "ceb_amr_only"
    cfg["data"]["train"]["augment_transform"] = [copy.deepcopy(AMR_BLOCK)]
    return cfg


def make_loso_patch(system, loso_dir):
    def patch(cfg):
        base = f"{loso_dir}/holdout_{system}"
        cfg["exp_name"] = f"ceb_loso_holdout_{system}"
        cfg["data"]["train"]["parquet_files"] = [f"{base}/train.parquet"]
        cfg["data"]["val"]["parquet_files"] = [f"{base}/val.parquet"]
        return cfg

    return patch


VARIANTS = {
    "noaug": {
        "patch": patch_noaug,
        "allowed": {"exp_name", "data.train.augment_transform"},
        "why": "ablation: what codec augmentation buys",
    },
    "opus": {
        "patch": patch_opus,
        "allowed": {"exp_name", "data.train.augment_transform"},
        "why": "cross-codec: train Opus, test AMR",
    },
    "amr": {
        "patch": patch_amr,
        "allowed": {"exp_name", "data.train.augment_transform"},
        "why": "cross-codec: train AMR, test Opus",
    },
}


# ---------------------------------------------------------------------------
# DIFFING
# ---------------------------------------------------------------------------


def flatten(obj, prefix=""):
    """Dotted-path view of a nested config, for exact comparison."""
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # Compare lists whole rather than per-index: an augmentation
        # block going from two entries to one should read as a single
        # change, not as three.
        out[prefix] = repr(obj)
    else:
        out[prefix] = obj

    return out


def diff(base, derived):
    """Dotted keys whose value differs, plus keys added or removed."""
    fb, fd = flatten(base), flatten(derived)
    changed = {}

    for key in sorted(set(fb) | set(fd)):
        before, after = fb.get(key, "<absent>"), fd.get(key, "<absent>")

        if before != after:
            changed[key] = (before, after)

    return changed


def top_level_of(key, allowed):
    """
    True when `key` is covered by an allowed path.

    A patch allowed to touch data.train.augment_transform may change
    anything beneath it, since removing the block deletes children.
    """
    return any(key == a or key.startswith(a + ".") for a in allowed)


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------


def build(name, spec, base_cfg, base_path):
    derived = spec["patch"](copy.deepcopy(base_cfg))
    changed = diff(base_cfg, derived)

    unexpected = {
        k: v for k, v in changed.items() if not top_level_of(k, spec["allowed"])
    }

    return derived, changed, unexpected


def render(name, spec, base_path, changed):
    """A header explaining exactly how this file differs from the base."""
    lines = [
        "# " + "-" * 73,
        f"# {name}.yaml -- GENERATED, do not edit by hand",
        "#",
        f"# Written by training/make_ablation_configs.py from {base_path}.",
        f"# Regenerate rather than editing: the point of this file is that",
        f"# it differs from the base in exactly the keys listed below and",
        f"# nowhere else, and the generator verifies that.",
        "#",
        f"# Purpose: {spec['why']}",
        "#",
        "# Differences from the base config:",
    ]

    for key, (before, after) in changed.items():
        lines.append(f"#   {key}")
        lines.append(f"#     base    : {before}")
        lines.append(f"#     this    : {after}")

    lines += [
        "#",
        "# Everything else -- seed, split, batch size, learning rate,",
        "# scheduler, early stopping -- is identical to the base. That is",
        "# what makes the comparison valid.",
        "#",
        "# Run:",
        f"#   python training\\run_training.py --config training\\configs\\{name}.yaml",
        "# " + "-" * 73,
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate experiment-matrix configs from a base config, "
            "verifying that each differs only in its declared keys."
        )
    )

    parser.add_argument(
        "--base",
        default="training/configs/train_main.yaml",
        help="config every variant is derived from",
    )
    parser.add_argument(
        "--out-dir",
        default="training/configs",
        help="where to write the generated configs",
    )
    parser.add_argument(
        "--loso-dir",
        default="manifests/loso",
        help=(
            "output of make_loso_splits.py. A loso variant is generated "
            "per holdout_* directory found there."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="generate only these variants",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write the files (default: dry run)",
    )

    args = parser.parse_args()

    base_path = Path(args.base)

    if not base_path.exists():
        print(f"ERROR: base config not found: {base_path}", file=sys.stderr)
        return 1

    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    variants = dict(VARIANTS)

    # LOSO variants are discovered, not hardcoded, so the matrix tracks
    # whatever make_loso_splits.py actually produced.
    loso_root = Path(args.loso_dir)
    found_loso = sorted(loso_root.glob("holdout_*")) if loso_root.is_dir() else []

    for d in found_loso:
        system = d.name[len("holdout_") :]
        variants[f"loso_{system}"] = {
            "patch": make_loso_patch(system, loso_root.as_posix()),
            "allowed": {
                "exp_name",
                "data.train.parquet_files",
                "data.val.parquet_files",
            },
            "why": f"leave-one-system-out: {system} never seen in training",
        }

    if not found_loso:
        print(
            f"NOTE: no holdout_* dirs under {loso_root} -- skipping LOSO"
            " variants. Run make_loso_splits.py first if you want them.\n"
        )

    if args.only:
        unknown = set(args.only) - set(variants)

        if unknown:
            print(
                f"ERROR: unknown variant(s) {sorted(unknown)}. "
                f"Available: {sorted(variants)}",
                file=sys.stderr,
            )
            return 1

        variants = {k: v for k, v in variants.items() if k in args.only}

    print("=" * 78)
    print("ABLATION CONFIGS" + ("" if args.apply else "  (DRY RUN)"))
    print("=" * 78)
    print()
    print(f"  base    : {base_path}")
    print(f"  out     : {args.out_dir}")
    print(f"  variants: {', '.join(variants)}")
    print()

    out_dir = Path(args.out_dir)
    failed = False
    written = []

    for name, spec in variants.items():
        derived, changed, unexpected = build(name, spec, base_cfg, base_path)

        print("-" * 78)
        print(f"  {name}.yaml   -- {spec['why']}")
        print("-" * 78)

        for key, (before, after) in changed.items():
            print(f"    {key}")
            print(f"      base : {before}")
            print(f"      this : {after}")

        if not changed:
            failed = True
            print("    FAIL: identical to the base -- the patch did nothing")
            print()
            continue

        if unexpected:
            failed = True
            print()
            for key, (before, after) in unexpected.items():
                print(
                    f"    FAIL: changed {key}, which is not declared in"
                    f" allowed={sorted(spec['allowed'])}",
                    file=sys.stderr,
                )
            print("    NOT WRITTEN")
            print()
            continue

        print(f"    OK: {len(changed)} declared change(s), nothing else moved")

        if args.apply:
            out_path = out_dir / f"{name}.yaml"
            out_dir.mkdir(parents=True, exist_ok=True)
            body = yaml.safe_dump(derived, sort_keys=False, width=100)
            out_path.write_text(
                render(name, spec, base_path, changed) + body,
                encoding="utf-8",
            )
            written.append(out_path)
            print(f"    written: {out_path}")

        print()

    print("=" * 78)

    if failed:
        print("RESULT: FAILED -- do not train on these.", file=sys.stderr)
        return 1

    if not args.apply:
        print("Dry run. Re-run with --apply to write the files.")
        return 0

    print(f"Wrote {len(written)} config(s).")
    print()
    print("Run them ONE AT A TIME -- each needs the whole GPU, and two")
    print("concurrent runs will OOM an 8 GB card:")
    print()

    for p in written:
        print(f"  python training\\run_training.py --config {p}")

    print()
    print("Budget ~2h10m per epoch; with patience 7 expect 24-48 h each.")
    print("Each checkpoint is ~3.8 GB and one is written per improvement,")
    print("so watch free disk between runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
