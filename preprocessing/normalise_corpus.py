r"""
normalise_corpus.py

Puts every clip in the corpus through one identical processing chain,
so that the class label stops being readable from production artefacts.

THE CHAIN

Every file, both classes, no exceptions -- that is the entire point:

    1. decode to mono float
    2. resample to 16 kHz with one resampler
    3. trim leading/trailing silence at a fixed relative threshold
    4. peak-normalise so max|x| = 1.0
    5. write 16 kHz mono PCM_16 WAV


LEVEL NORMALISATION IS PEAK, AND ONLY PEAK

ASVspoof 5's shortcut-artefact pipeline scales each waveform so its
peak amplitude equals 1.0, and that is what this script does. Gated
BS.1770-4 (LUFS) and plain RMS were both implemented and both removed:
each is defensible in the abstract, neither is what anti-spoofing
does, and choosing one means defending a departure from the literature
on your own measurements. Peak is the citable choice.

A common dither floor was implemented and removed for the same reason,
and because it backfired when measured: at -75 dBFS the noise sat
under half a least-significant bit, quiet passages still rounded to
bit-exact zero, and frac_exact_zero went from 0.5000 (clean) to 0.3115
(leaking). The dither created the cue it was meant to remove.

An MP3 encode/decode pass over every file, to give them all the same
codec history, was implemented and removed too. Measured on a paired
test, the residual MP3 difference after resampling to 16 kHz was
already weak (EER 0.4267) and equalising did not measurably improve it
(0.4000, inside the noise at that sample size). ASVspoof 2021 DF
applies codecs to create variability, not to remove it.

The rule behind all three removals: an unjustifiable preprocessing
step is worse than a small residual leak, because you have to defend
it and you cannot cite it.

WHAT ONE GAIN CANNOT FIX

A single gain per file can equalise exactly one level statistic. This
chain equalises PEAK, so `peak_db` goes to chance by construction --
but `rms_db` and `crest_db` are then free to vary with the waveform's
own peakiness, and either may still separate the classes afterwards.
Check them in the probe rather than assuming the gain fixed them.

That residual is not purely an artefact. TTS output really is more
dynamically compressed than a field recording, and crest factor is a
property of the signal, not only of the file. Some of it is a cue a
detector is entitled to use. If crest-related features dominate the
second probe run, the honest move is to report it as a limitation
rather than to flatten the waveform until it disappears -- over-
normalising destroys the vocoder artefacts the detector needs.

WHAT THIS DOES NOT TOUCH

Originals are never modified. Output goes to a parallel tree, so the
before/after probe comparison stays reproducible -- which is the
evidence you want in Section 4.10, not just the fixed number.

AFTER RUNNING

    1. rebuild the manifests against the new tree
    2. re-run probe_shortcuts.py and confirm the probe EER has moved
       toward 0.50
    3. only then retrain

Usage:

    # dry run (DEFAULT): sample processed in memory, nothing written
    python training\normalise_corpus.py --bonafide-root data\processed\bonafide --spoof-root meta-mms=data\processed\meta-mms --spoof-root elevenlabs=data\processed\elevenlabs --out-root data\normalised

    # do it -- the chain is the ASVspoof-5-aligned one
    python training\normalise_corpus.py --bonafide-root data\processed\bonafide --spoof-root meta-mms=data\processed\meta-mms --spoof-root elevenlabs=data\processed\elevenlabs --out-root data\normalised --apply --workers 6

"""

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_SR = 16000
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}

# Frame geometry for silence detection and level measurement.
FRAME_LEN = 400  # 25 ms @ 16 kHz
FRAME_HOP = 160  # 10 ms @ 16 kHz

# ---------------------------------------------------------------------------
# AUDIO CHAIN
# ---------------------------------------------------------------------------


def resample(x, sr_in, sr_out=TARGET_SR):
    if sr_in == sr_out:
        return x

    from scipy.signal import resample_poly

    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)

    return resample_poly(x, ratio.numerator, ratio.denominator)


def frame_rms(x):
    if len(x) < FRAME_LEN:
        return np.array([np.sqrt(np.mean(x**2) + 1e-20)])

    n = 1 + (len(x) - FRAME_LEN) // FRAME_HOP
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_HOP * np.arange(n)[:, None]

    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-20)


def trim_silence(x, top_db=40.0, margin_s=0.05):
    """
    Drop leading and trailing frames more than `top_db` below the
    file's own loudest frame.

    Relative rather than absolute, because an absolute threshold would
    behave differently on quiet recordings than on normalised TTS --
    which is the very asymmetry being removed. The margin keeps a
    little context so speech onsets are not clipped off.
    """
    if len(x) < FRAME_LEN:
        return x

    frames = frame_rms(x)
    peak = float(frames.max())

    if peak <= 0:
        return x

    keep = np.flatnonzero(frames >= peak * (10.0 ** (-top_db / 20.0)))

    if keep.size == 0:
        return x

    margin = int(margin_s * TARGET_SR)
    start = max(0, keep[0] * FRAME_HOP - margin)
    end = min(len(x), keep[-1] * FRAME_HOP + FRAME_LEN + margin)

    return x[start:end]


def normalise_level(x, peak_target):
    """
    Scale the waveform so its peak amplitude equals `peak_target`.

    This is the ASVspoof 5 shortcut-artefact operation. It is a single
    gain, so it cannot clip, and nothing is limited -- a limiter firing
    on some files and not others would put a level cue straight back
    in.

    Returns (audio, applied_gain_db).
    """
    peak_in = float(np.abs(x).max()) if x.size else 0.0

    if peak_in <= 0:
        return x.astype(np.float32), 0.0

    gain_db = 20.0 * np.log10(peak_target / peak_in)
    y = (x * (peak_target / peak_in)).astype(np.float32)

    return y, float(gain_db)


def process_one(job):
    """
    Full chain for one file. Runs inside a worker, so it returns a
    plain dict rather than raising.
    """
    import soundfile as sf

    src = job["src"]
    dst = job["dst"]

    result = {"src": src, "dst": dst, "source": job["source"], "ok": False}

    try:
        info = sf.info(src)
        result["native_sr"] = int(info.samplerate)
        result["native_subtype"] = str(info.subtype)

        x, sr = sf.read(src, dtype="float32", always_2d=False)

        if x.ndim > 1:
            x = x.mean(axis=1)

        result["in_samples"] = int(len(x))
        result["in_rms_db"] = 20.0 * np.log10(
            max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-10)
        )

        x = resample(np.asarray(x, dtype=np.float64), sr).astype(np.float32)

        x = trim_silence(x, job["top_db"])

        if len(x) < FRAME_LEN:
            result["error"] = "empty after trim"
            return result

        x, gain_db = normalise_level(x, job["peak_target"])

        result["gain_db"] = gain_db
        result["out_samples"] = int(len(x))
        result["out_rms_db"] = 20.0 * np.log10(
            max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-10)
        )

        if job["write"]:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, x, TARGET_SR, subtype="PCM_16")

        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# PLANNING
# ---------------------------------------------------------------------------


def build_jobs(roots, out_root, args, write):
    """One job per audio file, mirroring each input tree by relpath."""
    jobs = []
    counts = {}

    for source, root in roots:
        root_path = Path(root)

        if not root_path.is_dir():
            raise FileNotFoundError(f"{source}: {root} is not a directory")

        found = 0

        for src in sorted(root_path.rglob("*")):
            if not src.is_file():
                continue
            if src.suffix.lower() not in AUDIO_SUFFIXES:
                continue

            rel = src.relative_to(root_path).with_suffix(".wav")
            dst = Path(out_root) / source / rel

            jobs.append(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "source": source,
                    "top_db": args.top_db,
                    "peak_target": args.peak_target,
                    "write": write,
                }
            )
            found += 1

        counts[source] = found

    return jobs, counts


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------


def _run_batch(jobs):
    return [process_one(j) for j in jobs]


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_jobs(jobs, workers):
    """
    Execute, falling back to one process on memory pressure.

    Same failure mode as probe_shortcuts.py: every worker loads its own
    numpy/scipy, and on Windows that can exhaust the pagefile and
    surface as a bogus DLL import error.
    """
    total = len(jobs)
    results = []
    started = time.time()

    def progress():
        done = len(results)

        if done % 1000 and done != total:
            return

        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"  {done}/{total}  ({100.0 * done / total:.1f}%)"
            f"  {rate:.0f} files/s  ETA {eta / 60:.1f} min"
        )

    def serial():
        for job in jobs:
            results.append(process_one(job))
            progress()

    if workers <= 1:
        serial()
        return results

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_batch, b) for b in _chunk(jobs, 32)]

            for future in as_completed(futures):
                results.extend(future.result())
                progress()
    except (ImportError, OSError, MemoryError) as e:
        print()
        print("-" * 70)
        print(f"Worker pool failed: {type(e).__name__}: {e}")
        print(
            "\nAlmost always memory pressure rather than a code fault.\n"
            "Falling back to one process and restarting the pass.\n"
            "Pass --workers 1 next time to skip the restart."
        )
        print("-" * 70)
        print()
        results = []
        started = time.time()
        serial()

    return results


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------


def report(results, wrote, out_root):
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()
    print(f"  processed : {len(ok)}")
    print(f"  failed    : {len(bad)}")

    if bad:
        print()
        for r in bad[:10]:
            print(f"    {r.get('error')}  <- {r['src']}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")

    if not ok:
        return 1

    print()
    print("  LEVEL SPREAD BY SOURCE -- this is the shortcut being closed.")
    print("  Before, the classes sat at different levels. After, every")
    print("  source should share a near-identical mean and a standard")
    print("  deviation close to zero.")
    print()
    print(f"    {'source':<14} {'n':>7} {'in dBFS':>18} {'out dBFS':>18}")
    print("    " + "-" * 60)

    sources = sorted({r["source"] for r in ok})

    for source in sources:
        rows = [r for r in ok if r["source"] == source]
        a = np.array([r["in_rms_db"] for r in rows])
        b = np.array([r["out_rms_db"] for r in rows])
        print(
            f"    {source:<14} {len(rows):>7}"
            f"   {a.mean():>7.2f} +/- {a.std():<6.2f}"
            f"   {b.mean():>7.2f} +/- {b.std():<6.2f}"
        )

    spread = np.std(
        [
            np.mean([r["out_rms_db"] for r in ok if r["source"] == s])
            for s in sources
        ]
    )

    print()
    print(f"    between-source spread after: {spread:.3f} dB")

    if spread < 0.5:
        print("    -> levels now match across sources.")
    else:
        print(
            "    -> WARNING: sources still differ. Check the failures\n"
            "       above and whether one source is mostly silence."
        )

    dur = sum(r["out_samples"] for r in ok) / TARGET_SR / 3600.0
    print()
    print(f"  total audio out: {dur:.2f} h")

    if wrote:
        print(f"  written to: {out_root}")

    return 0


def next_steps(out_root, roots):
    spoof = " ".join(
        f"--spoof-root {s}={Path(out_root) / s}"
        for s, _ in roots
        if s != "bonafide"
    )

    print()
    print("=" * 70)
    print("NEXT")
    print("=" * 70)
    print()
    print("  1. Rebuild the manifests against the normalised tree:")
    print()
    print(
        "     python training\\build_manifests.py "
        f"--bonafide-root {Path(out_root) / 'bonafide'} {spoof} "
        "--speaker-manifest manifests\\manifest_bonafide.csv "
        "--out manifests\\splits_norm"
    )
    print()
    print("     Filenames keep their stems, so the speaker manifest")
    print("     still matches. Extensions all become .wav.")
    print()
    print("  2. Re-run the probe against the new splits:")
    print()
    print(
        "     python training\\probe_shortcuts.py --workers 1 "
        "--splits-dir manifests\\splits_norm "
        "--cache outputs\\probe_normalised.parquet"
    )
    print()
    print("     Compare against the old run. You want the probe EER to")
    print("     move from 0.0250 toward 0.50. Keep BOTH reports -- the")
    print("     before/after pair is the evidence for Section 4.10, and")
    print("     it is worth more than the fixed number on its own.")
    print()
    print("  3. Point the training configs at manifests/splits_norm")
    print("     and only then start the real runs.")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def parse_spoof_root(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--spoof-root expects name=path, got {value!r}"
        )

    name, path = value.split("=", 1)

    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalise every clip through one identical chain so the "
            "class label stops being readable from file production."
        )
    )

    parser.add_argument("--bonafide-root")
    parser.add_argument(
        "--spoof-root",
        action="append",
        type=parse_spoof_root,
        default=[],
        metavar="NAME=PATH",
        help="repeatable, e.g. --spoof-root elevenlabs=data\\elevenlabs",
    )
    parser.add_argument("--out-root", default="data/normalised")

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "actually write files. Without it this is a dry run: a "
            "sample is processed in memory and the before/after levels "
            "are reported, but nothing is written."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=400,
        help="clips per source to process in a dry run (default: 400)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "worker processes (default: 2). Use 1 while training is "
            "running; each worker loads its own numpy/scipy."
        ),
    )

    parser.add_argument(
        "--peak-target",
        type=float,
        default=1.0,
        help=(
            "peak amplitude every waveform is scaled to "
            "(default: 1.0, the ASVspoof 5 value)"
        ),
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=40.0,
        help="silence trim threshold below peak frame (default: 40)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="write the per-file record to this JSON file",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    roots = []

    if args.bonafide_root:
        roots.append(("bonafide", args.bonafide_root))

    roots.extend(args.spoof_root)

    if not roots:
        parser.print_help()
        print(
            "\nNothing to do: pass --bonafide-root and/or --spoof-root.",
            file=sys.stderr,
        )
        return 2

    try:
        jobs, counts = build_jobs(roots, args.out_root, args, args.apply)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("NORMALISE CORPUS" + ("" if args.apply else "  (DRY RUN)"))
    print("=" * 70)
    print()

    for source, root in roots:
        print(f"  {source:<14} {counts[source]:>7} files   <- {root}")

    print()
    print(
        f"  chain        : 16 kHz mono -> trim -> "
        f"peak -> {args.peak_target:g}"
    )
    print(f"  output       : {args.out_root}")
    print(f"  workers      : {args.workers}")
    print()

    if not jobs:
        print("ERROR: no audio files found.", file=sys.stderr)
        return 1

    if not args.apply:
        rng = np.random.default_rng(args.seed)
        picked = []

        for source in counts:
            pool = [j for j in jobs if j["source"] == source]
            take = min(args.sample, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            picked.extend(pool[i] for i in idx)

        print(
            f"  DRY RUN: processing {len(picked)} sampled clips in "
            "memory. Nothing will be written."
        )
        print()
        jobs = picked

    results = run_jobs(jobs, args.workers)

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.report_json}")

    code = report(results, args.apply, args.out_root)

    if code == 0 and args.apply:
        next_steps(args.out_root, roots)
    elif code == 0:
        print()
        print("  Dry run only -- nothing written. Re-run with --apply")
        print("  once the level columns above look right.")
        print()

    return code


if __name__ == "__main__":
    sys.exit(main())
