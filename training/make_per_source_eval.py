r"""
make_per_source_eval.py

Builds everything needed to answer two questions the pooled test EER
cannot:

  1. PER-SOURCE EER -- is the detector equally good at both synthesis
     systems, or is one of them carrying the result?

  2. CALIBRATED METRICS -- accuracy, precision, recall, F1 and AUC at a
     threshold taken from validation rather than from the test set.

CONFIGS

  test_per_source.yaml -> both per-source parquets, reported separately
                          via dataset_names
  test_val.yaml        -> the VALIDATION split, so score_predictions.py
                          has a file to calibrate its threshold on

Usage:

    python training\make_per_source_eval.py            # dry run
    python training\make_per_source_eval.py --apply
"""

import argparse
import copy
import sys
from pathlib import Path

try:
    import pandas as pd
    import yaml
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow pyyaml",
          file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# CROSS-CONDITION CODEC BLOCKS
# ---------------------------------------------------------------------------
#
# noise_ratio is 1.0 here, NOT the 0.5 used in training. During training
# half the batch is degraded so the model sees both conditions. A test
# CONDITION is a single condition: every clip must go through the codec,
# or the reported number is an average of clean and degraded and means
# neither.

OPUS_TEST = {
    "type": "opus_codec",
    "noise_ratio": 1.0,
    "bitrate_min": 6000,
    "bitrate_max": 32000,
}

AMR_TEST = {
    "type": "amrnb_codec",
    "noise_ratio": 1.0,
    "bitrates": [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200],
}

LABEL_BONAFIDE = "bonafide"
LABEL_SPOOF = "spoof"


# ---------------------------------------------------------------------------
# DIFF -- same contract as make_ablation_configs.py
# ---------------------------------------------------------------------------


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = repr(obj)
    else:
        out[prefix] = obj
    return out


def diff(base, derived):
    fb, fd = flatten(base), flatten(derived)
    changed = {}
    for key in sorted(set(fb) | set(fd)):
        before, after = fb.get(key, "<absent>"), fd.get(key, "<absent>")
        if before != after:
            changed[key] = (before, after)
    return changed


def covered(key, allowed):
    return any(key == a or key.startswith(a + ".") for a in allowed)


# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------


def split_by_source(test_path: Path, out_dir: Path, apply: bool):
    df = pd.read_parquet(test_path)

    bona = df[df["label"] == LABEL_BONAFIDE]
    spoof = df[df["label"] == LABEL_SPOOF]

    if bona.empty or spoof.empty:
        print("ERROR: test split has no bonafide or no spoof rows.",
              file=sys.stderr)
        sys.exit(1)

    systems = sorted(spoof["spoof_method"].unique())
    print(f"  bonafide pool : {len(bona)} clips, "
          f"{bona['speaker_id'].nunique()} speakers")
    print(f"  spoof systems : {', '.join(systems)}")
    print()

    written = {}

    for system in systems:
        sub = spoof[spoof["spoof_method"] == system]
        merged = pd.concat([bona, sub], ignore_index=True)
        out = out_dir / f"test_{system}.parquet"

        print(f"  test_{system}.parquet")
        print(f"      bonafide {len(bona):6}   spoof {len(sub):6}"
              f"   total {len(merged):6}")

        if apply:
            merged.to_parquet(out, index=False)
            print(f"      written: {out}")

        written[system] = out
        print()

    return systems, written


# ---------------------------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------------------------


def render_header(name, base_path, why, changed):
    lines = [
        "# " + "-" * 73,
        f"# {name}.yaml -- GENERATED, do not edit by hand",
        "#",
        f"# Written by training/make_per_source_eval.py from {base_path}.",
        "# It differs from the base in exactly the keys below and nowhere",
        "# else, and the generator verifies that -- the model/loss blocks",
        "# must match the checkpoint or it will not load.",
        "#",
        f"# Purpose: {why}",
        "#",
        "# Differences from the base config:",
    ]
    for key, (before, after) in changed.items():
        lines.append(f"#   {key}")
        lines.append(f"#     base : {before}")
        lines.append(f"#     this : {after}")
    lines += [
        "#",
        "# Run:",
        f"#   deepfense test --config training\\configs\\{name}.yaml \\",
        "#       --checkpoint outputs\\<run>\\best_model.pth",
        "#",
        "# Predictions land next to the CHECKPOINT, not next to this file:",
        "#   <checkpoint dir>\\results\\predictions\\<dataset>_predictions.txt",
        "# " + "-" * 73,
        "",
    ]
    return "\n".join(lines)


def build_config(name, base_cfg, base_path, patch, why, out_dir, apply):
    derived = patch(copy.deepcopy(base_cfg))
    changed = diff(base_cfg, derived)
    allowed = {"data.test"}
    unexpected = {k: v for k, v in changed.items() if not covered(k, allowed)}

    print(f"  {name}.yaml   -- {why}")
    for key, (before, after) in changed.items():
        print(f"      {key}")
        print(f"        base : {before}")
        print(f"        this : {after}")

    if not changed:
        print("      FAIL: identical to base -- patch did nothing\n")
        return False

    if unexpected:
        for key in unexpected:
            print(f"      FAIL: changed {key}, outside data.test",
                  file=sys.stderr)
        print("      NOT WRITTEN\n")
        return False

    print(f"      OK: {len(changed)} declared change(s), nothing else moved")

    if apply:
        path = out_dir / f"{name}.yaml"
        body = yaml.safe_dump(derived, sort_keys=False, width=100)
        path.write_text(
            render_header(name, base_path, why, changed) + body,
            encoding="utf-8",
        )
        print(f"      written: {path}")

    print()
    return True


def main():
    ap = argparse.ArgumentParser(
        description=("Build per-source test parquets and the evaluation "
                     "configs derived from the training config.")
    )
    ap.add_argument("--splits-dir", default="manifests/splits_norm",
                    help="directory holding test.parquet and val.parquet")
    ap.add_argument("--base", default="training/configs/train_main.yaml",
                    help="training config the eval configs are derived from")
    ap.add_argument("--config-dir", default="training/configs")
    ap.add_argument("--loso-dir", default="manifests/loso",
                    help=("output of make_loso_splits.py. One test config "
                          "is generated per holdout_* directory found."))
    ap.add_argument("--apply", action="store_true",
                    help="actually write the files (default: dry run)")
    args = ap.parse_args()

    splits = Path(args.splits_dir)
    base_path = Path(args.base)
    cfg_dir = Path(args.config_dir)

    for p in (splits / "test.parquet", splits / "val.parquet", base_path):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("=" * 78)
    print("PER-SOURCE EVALUATION" + ("" if args.apply else "  (DRY RUN)"))
    print("=" * 78)
    print()

    systems, written = split_by_source(
        splits / "test.parquet", splits, args.apply
    )

    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    if "data" not in base_cfg or "test" not in base_cfg["data"]:
        print("ERROR: base config has no data.test block to derive from.",
              file=sys.stderr)
        sys.exit(1)

    def patch_per_source(cfg):
        cfg["data"]["test"]["parquet_files"] = [
            (splits / f"test_{s}.parquet").as_posix() for s in systems
        ]
        cfg["data"]["test"]["dataset_names"] = list(systems)
        return cfg

    def patch_val(cfg):
        cfg["data"]["test"]["parquet_files"] = [
            (splits / "val.parquet").as_posix()
        ]
        cfg["data"]["test"]["dataset_names"] = ["val"]
        return cfg

    def _codec_patch(block, name):
        def patch(cfg):
            cfg["data"]["test"]["parquet_files"] = [
                (splits / "test.parquet").as_posix()
            ]
            cfg["data"]["test"]["dataset_names"] = [name]
            cfg["data"]["test"]["augment_transform"] = [copy.deepcopy(block)]
            return cfg
        return patch

    print("-" * 78)
    ok = True
    ok &= build_config(
        "test_per_source", base_cfg, base_path, patch_per_source,
        "per-source EER: one number per synthesis system, shared bonafide",
        cfg_dir, args.apply,
    )
    ok &= build_config(
        "test_val", base_cfg, base_path, patch_val,
        "validation scores, so score_predictions.py can calibrate a "
        "threshold off the test set",
        cfg_dir, args.apply,
    )

    ok &= build_config(
        "test_opus", base_cfg, base_path,
        _codec_patch(OPUS_TEST, "test_opus"),
        "cross-condition: full test set through Opus 6-32 kbps",
        cfg_dir, args.apply,
    )
    ok &= build_config(
        "test_amr", base_cfg, base_path,
        _codec_patch(AMR_TEST, "test_amr"),
        "cross-condition: full test set through AMR-NB",
        cfg_dir, args.apply,
    )

    # -----------------------------------------------------------------
    # LOSO test configs
    # -----------------------------------------------------------------
    #
    # BOTH test sets go in ONE config, reported separately via
    # dataset_names. The LOSO result is not either number on its own --
    # it is the GAP between them for the same checkpoint:
    #
    #     EER(unseen) - EER(matched)
    #
    # Scoring them in separate runs would be two checkpoint loads and
    # an invitation to compare numbers from different runs by mistake.

    loso_root = Path(args.loso_dir)
    found = sorted(loso_root.glob("holdout_*")) if loso_root.is_dir() else []

    if not found:
        print(f"NOTE: no holdout_* under {loso_root} -- skipping LOSO test"
              " configs. Run make_loso_splits.py first.\n")

    def _loso_patch(d, held, trained_on):
        def patch(cfg):
            cfg["data"]["test"]["parquet_files"] = [
                (d / "test_matched.parquet").as_posix(),
                (d / "test_unseen.parquet").as_posix(),
            ]
            cfg["data"]["test"]["dataset_names"] = [
                f"matched_{trained_on}", f"unseen_{held}",
            ]
            cfg["data"]["test"].pop("augment_transform", None)
            return cfg
        return patch

    for d in found:
        held = d.name[len("holdout_"):]
        trained_on = "+".join(s for s in systems if s != held) or "other"
        ok &= build_config(
            f"test_{d.name}", base_cfg, base_path,
            _loso_patch(d, held, trained_on),
            f"LOSO: matched ({trained_on}) vs unseen ({held}), one pass",
            cfg_dir, args.apply,
        )

    print("=" * 78)
    if not ok:
        print("RESULT: FAILED -- nothing usable written.", file=sys.stderr)
        return 1
    if not args.apply:
        print("Dry run. Re-run with --apply to write.")
        return 0

    print("Next, with CKPT = your best_model.pth:\n")
    print("  deepfense test --config training\\configs\\test_val.yaml "
          "--checkpoint %CKPT%")
    print("  deepfense test --config training\\configs\\test_per_source.yaml "
          "--checkpoint %CKPT%")
    print()
    print("Then, from the predictions written beside the checkpoint:\n")
    print("  python training\\score_predictions.py ^")
    print("      --calibrate <ckptdir>\\results\\predictions\\val_predictions.txt ^")
    for s in systems:
        print(f"      --evaluate <ckptdir>\\results\\predictions\\{s}_predictions.txt ^")
    print("      --bootstrap 2000 --csv outputs\\per_source_metrics.csv")
    print()
    print("Read the GAP between the systems, not the point estimates --")
    print("the spoof counts differ, so use the bootstrap intervals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
