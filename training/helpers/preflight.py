r"""
preflight.py

Checks that should pass BEFORE committing 18-22 hours of GPU time.

Every check here exists because its absence already cost this project
real time:

  * manifests/splits_norm was silently overwritten with VC data and nine
    configs kept reading it for two days. Clip IDs are identical across
    the two corpora, so nothing errored -- the numbers were simply wrong.
  * The LOSO manifests were count-balanced but only ~48% clip-paired,
    which quietly removed the content control from one experiment while
    leaving it in the other.
  * A run died partway through because the disk filled with checkpoints.
  * A checkpoint was truncated to 556 MB by a kill during a save.
  * `deepfense train` fails at epoch 0 with KeyError: 'opus_codec'
    because the custom transforms are never imported by the bare CLI.

Usage:

    python training\preflight.py --config training\configs_vc\train_vc.yaml
    python training\preflight.py --config training\configs_vc\noaug.yaml --expect-corpus vc
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
    import yaml
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow pyyaml", file=sys.stderr)
    sys.exit(1)

SUFFIX = re.compile(r"\.\d$")
GREEN, RED, YEL = "PASS", "FAIL", "WARN"
problems, warnings_ = [], []


def say(status, label, detail=""):
    print(f"  {status}  {label}")
    if detail:
        for line in detail.rstrip().splitlines():
            print(f"          {line}")
    if status == RED:
        problems.append(label)
    elif status == YEL:
        warnings_.append(label)


def head(t):
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def check_manifest(path: Path, role: str):
    """Counts, systems, clip-level pairing, and the audio root."""
    if not path.exists():
        say(RED, f"{role}: manifest missing", str(path))
        return None

    df = pd.read_parquet(path)
    bona = set(df[df.label == "bonafide"]["ID"])
    spoof = df[df.label == "spoof"]
    systems = dict(spoof["spoof_method"].value_counts())

    paired = spoof["ID"].map(lambda s: SUFFIX.sub("", s)).isin(bona).sum()
    pct = 100.0 * paired / len(spoof) if len(spoof) else 0.0

    root = ""
    if len(spoof) and "path" in df.columns:
        p = str(spoof["path"].iloc[0])
        for marker in ("normalised", "processed"):
            if marker in p:
                root = p.split(marker)[0] + marker
                break

    detail = (f"{len(df)} clips | bonafide {len(bona)} | spoof {len(spoof)} {systems}\n"
              f"audio root: {root or '(unknown)'}")

    say(GREEN, f"{role}: {path.name} loaded", detail)

    if len(spoof):
        if pct >= 99.9:
            say(GREEN, f"{role}: clip-level pairing 100%")
        elif pct >= 50:
            say(YEL, f"{role}: pairing only {pct:.1f}%",
                f"{len(spoof) - paired} spoof clips have no bona fide twin. "
                "The content control is weakened.")
        else:
            say(RED, f"{role}: pairing only {pct:.1f}%",
                "The paired design is effectively absent here.")

    return set(df["speaker_id"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-run sanity checks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-corpus", choices=["vc", "mms"],
                    help="refuse if the manifests are not this corpus")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    head(f"PREFLIGHT  {cfg_path}")

    if not cfg_path.exists():
        say(RED, "config not found", str(cfg_path))
        return 1

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    say(GREEN, "config parses", f"exp_name: {cfg.get('exp_name')}  seed: {cfg.get('seed')}")

    # ---------------- manifests ----------------
    head("MANIFESTS  -- which corpus is this actually going to read?")

    speakers, all_methods = {}, set()
    for split in ("train", "val", "test"):
        d = cfg.get("data", {}).get(split, {})
        files = d.get("parquet_files") or []
        for f in files:
            p = Path(f)
            print(f"\n  [{split}] {f}")
            spk = check_manifest(p, split)
            if spk is not None:
                speakers.setdefault(split, set()).update(spk)
                df = pd.read_parquet(p)
                all_methods |= set(df[df.label == "spoof"]["spoof_method"].unique())

    print()
    if "meta-mms" in all_methods and "voice-conv" in all_methods:
        say(RED, "BOTH meta-mms and voice-conv present -- manifests are mixed")
    elif "voice-conv" in all_methods:
        say(GREEN, "corpus: VC (ElevenLabs + FreeVC)")
        if args.expect_corpus == "mms":
            say(RED, "corpus mismatch: expected MMS, found VC")
    elif "meta-mms" in all_methods:
        say(GREEN, "corpus: MMS (ElevenLabs + Meta MMS)")
        if args.expect_corpus == "vc":
            say(RED, "corpus mismatch: expected VC, found MMS")
    else:
        say(YEL, f"could not identify corpus from methods {all_methods}")

    # ---------------- leakage ----------------
    head("SPEAKER LEAKAGE  -- must be zero between train / val / test")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if a in speakers and b in speakers:
            n = len(speakers[a] & speakers[b])
            say(GREEN if n == 0 else RED, f"{a} vs {b}: {n} shared speaker(s)")

    # ---------------- augmentation ----------------
    head("AUGMENTATION")
    aug = cfg.get("data", {}).get("train", {}).get("augment_transform") or []
    types = [a.get("type") for a in aug]
    say(GREEN, f"train augment_transform: {types or 'NONE (ablation)'}")

    if any("codec" in str(t) for t in types):
        say(YEL, "custom codec transforms in use",
            "Launch with training\\run_training.py, NOT `deepfense train` --\n"
            "the bare CLI never imports training/codec_opus_amr.py and dies\n"
            "with KeyError: 'opus_codec'.")
        if shutil.which("ffmpeg"):
            say(GREEN, "ffmpeg found on PATH")
            try:
                r = subprocess.run([sys.executable, "training/codec_opus_amr.py",
                                    "--check-ffmpeg"],
                                   capture_output=True, text=True, timeout=60)
                tail = "\n".join(r.stdout.strip().splitlines()[-6:])
                say(GREEN if r.returncode == 0 else RED,
                    "ffmpeg encoder check", tail)
            except Exception as e:
                say(YEL, "encoder check did not run", str(e))
        else:
            say(RED, "ffmpeg NOT on PATH -- codec augmentation cannot run")

    # ---------------- disk ----------------
    head("RESOURCES")
    out = Path(cfg.get("output_dir", "./outputs"))
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    say(GREEN if free_gb >= args.min_free_gb else RED,
        f"free disk at {out}: {free_gb:.1f} GB",
        f"a full run writes 15-25 GB of checkpoints "
        f"(one per validation improvement, ~3.79 GB each)")

    # existing checkpoints that look truncated
    bad = []
    for p in Path("outputs").glob("*/best_model.pth"):
        sz = p.stat().st_size
        if sz < 3_000_000_000:
            bad.append(f"{p}  {sz} bytes")
    if bad:
        say(YEL, "truncated checkpoint(s) from earlier runs", "\n".join(bad))
    else:
        say(GREEN, "no truncated checkpoints in outputs/")

    # ---------------- gpu ----------------
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(0)
            say(GREEN if free / 1e9 > 5 else RED,
                f"GPU: {name}",
                f"{free/1e9:.1f} GB free of {total/1e9:.1f} GB "
                f"(another training run still holding memory?)")
        else:
            say(RED, "CUDA not available")
    except Exception as e:
        say(YEL, "GPU check skipped", str(e))

    # ---------------- verdict ----------------
    head("VERDICT")
    if problems:
        print("  DO NOT START. Problems:")
        for p in problems:
            print(f"    - {p}")
        return 1
    if warnings_:
        print("  OK to start, but read these first:")
        for w in warnings_:
            print(f"    - {w}")
        return 0
    print("  All checks passed. Safe to start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
