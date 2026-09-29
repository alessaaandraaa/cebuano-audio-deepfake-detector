r"""
make_unseen_vc_eval.py

Builds an evaluation set for a voice-conversion system that was NEVER
used to build the training corpus, and a test config that scores an
existing checkpoint against it.

WHY

The trained corpus uses ONE voice-conversion system (FreeVC). Its
per-source test EER is 0.0002, against 0.0142 for ElevenLabs -- a 71x
gap. There are two explanations and they look identical in that number:

  benign  FreeVC is one fixed checkpoint, so every clip carries the same
          HiFi-GAN decoder signature. A seen, deterministic,
          single-generator attack IS near-trivial. That is what "seen
          attack" means.

  malign  the detector learned "FreeVC fingerprint = spoof, absence of
          it = real", which generalises to nothing.

The corpus cannot tell these apart, because it contains no other VC
system. This script builds the missing measurement: the same detector,
scored on a DIFFERENT voice-conversion system it has never seen.

    voice-conv (seen)    -> what you already have
    knnvc      (unseen)  -> what this produces

The second number is what tells a reader what the first one is worth.

WHAT IT DOES NOT CHANGE

Nothing. It reads manifests/splits_vc/test.parquet, keeps its bona fide
rows exactly as they are, and adds spoof rows pointing at the new audio.
The training corpus, the splits and every existing config are untouched,
so the run you already finished stays valid and comparable.

THE BONA FIDE POOL

By default the FULL test bona fide pool is used -- the same one
make_per_source_eval.py gives every other per-source file -- so this
number sits in the same table as the others without an asterisk. That
makes the classes unbalanced when the spoof side is a --limit subset.
EER is rank-based and unbiased by imbalance; it only costs resolution
on the spoof side, which the summary reports so you can judge it.

--balance subsamples the bona fide pool to match instead. Use it only
if you need a balanced ACC, and say so, because the pool then differs
from the other per-source evaluations.

PIPELINE

    # 0. knnvc has never run end to end -- audition 4 clips first
    python preprocessing\vc_convert.py --audition --models knnvc ^
        --split test --audition-pairs 2 --audition-clips 2

    # 1. generate, TEST speakers only, into a separate tree
    python preprocessing\vc_convert.py --model knnvc --apply ^
        --split test --limit 20 --out-root data\processed\vc-knnvc

    # 2. same normalisation chain as every other class
    python training\normalise_corpus.py ^
        --spoof-root knnvc=data\processed\vc-knnvc ^
        --out-root data\normalised --apply --workers 6

    # 3. this script
    python training\make_unseen_vc_eval.py --vc-root data\normalised\knnvc ^
        --method-name knnvc --apply

    # 4. score the checkpoint you already trained
    python training\run_training.py --test ^
        --config training\configs_vc\test_unseen_knnvc.yaml ^
        --checkpoint outputs\ceb_xlsr_aasist_vc_<stamp>\best_model.pth

READING THE RESULT

    ~0.3 - 0.5   the fingerprint is FreeVC-specific. The detector does
                 not do voice conversion, it does FreeVC. A measured,
                 reportable limitation -- and a reason to rebuild the
                 VC half from several generators.

    ~0.01 - 0.1  partial transfer. Plausible: FreeVC and kNN-VC both use
                 HiFi-GAN-family vocoders, so a shared waveform-generator
                 artefact would carry across even though the conversion
                 methods are unrelated.

    < 0.01       transfers across VC systems. The strongest outcome, and
                 it predicts that a NON-neural-vocoder VC (the ASVspoof
                 A17 waveform-filtering family) is where it would break.
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import pandas as pd
    import yaml
except ImportError:
    print("Missing dependency(s). Run: pip install pandas pyarrow pyyaml",
          file=sys.stderr)
    sys.exit(1)

SPOOF_SUFFIX = ".1"


def build_rows(test_df: pd.DataFrame, vc_root: Path, method: str):
    """Spoof rows for every bona fide clip that has converted audio."""
    bona = test_df[test_df["spoof_method"] == "none"].copy()
    rows, missing = [], 0

    for r in bona.itertuples(index=False):
        f = vc_root / r.speaker_id / f"{r.ID}{SPOOF_SUFFIX}.wav"
        if not f.exists() or f.stat().st_size < 1024:
            missing += 1
            continue
        rows.append({
            "ID": f"{r.ID}{SPOOF_SUFFIX}",
            "path": str(f.resolve()),
            "label": "spoof",
            "speaker_id": r.speaker_id,
            "spoof_method": method,
        })
    return bona, pd.DataFrame(rows), missing


def check(bona: pd.DataFrame, spoof: pd.DataFrame, train_df: pd.DataFrame,
          method: str):
    """Everything that would silently invalidate this number."""
    problems = []

    if spoof.empty:
        problems.append("no converted audio found at all")

    seen = set(train_df["spoof_method"].unique()) - {"none"}
    if method in seen:
        problems.append(
            f"'{method}' IS in the training manifest -- it is not unseen, "
            f"and this measurement is meaningless. Training systems: "
            f"{sorted(seen)}")

    if not spoof.empty:
        overlap = set(spoof["speaker_id"]) & set(train_df["speaker_id"])
        if overlap:
            problems.append(
                f"{len(overlap)} speaker(s) also appear in TRAIN: "
                f"{sorted(overlap)[:5]}")

        stems = {re.sub(r"[.]\d$", "", i) for i in spoof["ID"]}
        if not stems <= set(bona["ID"]):
            problems.append("some spoof clips have no bona fide counterpart "
                            "in the test split")

    checks = [
        ("converted audio present", not spoof.empty),
        (f"'{method}' never seen in training", method not in seen),
        ("no speaker shared with the training split",
         spoof.empty or not (set(spoof["speaker_id"]) &
                             set(train_df["speaker_id"]))),
        ("every spoof clip pairs to a test bona fide clip",
         spoof.empty or {re.sub(r"[.]\d$", "", i)
                         for i in spoof["ID"]} <= set(bona["ID"])),
    ]
    return checks, problems


def write_config(base: Path, out_cfg: Path, parquet: Path, method: str):
    """Derive the test config, changing only what must change."""
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    ref = yaml.safe_load(base.read_text(encoding="utf-8"))

    cfg["exp_name"] = f"ceb_eval_unseen_{method}"
    cfg["data"]["test"]["parquet_files"] = [parquet.as_posix()]
    cfg["data"]["test"]["dataset_names"] = [method]

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            key = f"{p}.{k}" if p else k
            o.update(flat(v, key)) if isinstance(v, dict) else o.update({key: repr(v)})
        return o

    fa, fb = flat(ref), flat(cfg)
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    allowed = {"exp_name", "data.test.parquet_files", "data.test.dataset_names"}
    extra = set(diff) - allowed
    if extra:
        print(f"ERROR: config would change unexpected keys: {extra}",
              file=sys.stderr)
        sys.exit(1)

    header = (
        f"# ----------------------------------------------------------\n"
        f"# test_unseen_{method}.yaml -- GENERATED by "
        f"make_unseen_vc_eval.py\n"
        f"#\n"
        f"# Scores an existing checkpoint against a voice-conversion\n"
        f"# system that was never used to build the training corpus.\n"
        f"# Differs from the base config in exactly three keys:\n"
        f"#   exp_name, data.test.parquet_files, data.test.dataset_names\n"
        f"# ----------------------------------------------------------\n")
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(
        header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8")
    return diff


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build an unseen-VC-system evaluation set.")
    ap.add_argument("--splits-dir", default="manifests/splits_vc")
    ap.add_argument("--vc-root", default="data/normalised/knnvc",
                    help="normalised audio for the unseen system")
    ap.add_argument("--method-name", default="knnvc")
    ap.add_argument("--out", default="manifests/eval_unseen_vc")
    ap.add_argument("--base", default="training/configs_vc/train_vc.yaml")
    ap.add_argument("--config-dir", default="training/configs_vc")
    ap.add_argument("--balance", action="store_true",
                    help="subsample bona fide to match the spoof count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    splits = Path(args.splits_dir)
    vc_root = Path(args.vc_root)
    method = args.method_name

    if not vc_root.is_dir():
        print(f"ERROR: no audio at {vc_root}\n"
              f"       Generate and normalise it first -- see the module "
              f"docstring.", file=sys.stderr)
        return 1

    test_df = pd.read_parquet(splits / "test.parquet")
    train_df = pd.read_parquet(splits / "train.parquet")

    bona, spoof, missing = build_rows(test_df, vc_root, method)

    print()
    print(f"  unseen system   {method}")
    print(f"  audio           {vc_root}")
    print(f"  test bona fide  {len(bona)}")
    print(f"  converted       {len(spoof)}   ({missing} bona fide clips have "
          f"no counterpart)")
    print()

    checks, problems = check(bona, spoof, train_df, method)
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    print()

    if problems:
        print("REFUSING:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if args.balance and len(spoof) < len(bona):
        bona = bona.sample(n=len(spoof), random_state=args.seed)
        print(f"  balanced: bona fide subsampled to {len(bona)}\n")

    out_df = pd.concat([bona, spoof], ignore_index=True)
    n_sp = len(spoof)
    print(f"  eval set        {len(out_df)} rows "
          f"({len(bona)} bonafide / {n_sp} spoof)")
    print(f"  EER resolution  ~{1 / n_sp:.4f} on the spoof side "
          f"({n_sp} clips)")
    if n_sp < 200:
        print("  WARNING: fewer than 200 spoof clips. Fine for telling 0.002 "
              "from 0.3,\n           too coarse to quote to four decimals.")
    print()

    parquet = Path(args.out) / f"test_unseen_{method}.parquet"
    out_cfg = Path(args.config_dir) / f"test_unseen_{method}.yaml"

    if not args.apply:
        print(f"  would write  {parquet}")
        print(f"  would write  {out_cfg}")
        print("\n  Dry run. Add --apply.")
        return 0

    parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet, index=False)
    diff = write_config(Path(args.base), out_cfg, parquet, method)

    print(f"  wrote  {parquet}")
    print(f"  wrote  {out_cfg}   (differs from base in {len(diff)} keys: "
          f"{', '.join(diff)})")
    print("\n  Next:")
    print(f"    python training\\run_training.py --test "
          f"--config {out_cfg} --checkpoint outputs\\<run>\\best_model.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
