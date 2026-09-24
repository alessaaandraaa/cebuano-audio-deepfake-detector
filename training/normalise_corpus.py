r"""
normalise_corpus.py

Puts every clip in the corpus through one identical processing chain,
so that the class label stops being readable from production artefacts.

WHY THIS EXISTS -- the measured problem

probe_shortcuts.py, run on the 70/15/15 splits, returned:

    Probe EER  0.0250        (logistic regression, 13 summary stats)
    Model EER  0.0435        (XLS-R 300M + AASIST, 3 epochs)

A model that cannot hear words beat the detector. The probe also named
the culprits, and they are not subtle:

    FILE PROVENANCE
        4041 / 8000 sampled spoof clips are MP3 @ 44.1 kHz
           0 / 8000 bonafide clips are
        -> ElevenLabs returns MP3 by default; Meta MMS and the
           bonafide corpus are both 16 kHz PCM WAV

    SINGLE-FEATURE SEPARABILITY (train)
        rms_db            EER 0.0680   loudness alone
        trail_silence_s   EER 0.0940   trailing quiet tail
        peak_db           EER 0.1565
        crest_db          EER 0.2010
        dyn_range_db      EER 0.2788

Loudness is the dominant leak. The TTS systems normalise their output
and field recordings do not, so "how loud is this file" is very nearly
a label. Bandwidth, by contrast, is a non-issue here:

        band_7k_8k_ratio  EER 0.4380   (chance)
        hf_ratio_7k       EER 0.4387   (chance)
        frac_exact_zero   EER 0.5000   (chance exactly)

so this script does NOT lowpass anything. It fixes what is measurably
broken and leaves the rest alone.

THE CHAIN

Every file, both classes, no exceptions -- that is the entire point:

    1. decode to mono float
    2. resample to 16 kHz with one resampler
    3. trim leading/trailing silence at a fixed relative threshold
    4. MP3 encode/decode round trip at fixed settings
    5. trim again (the encoder adds its own padding)
    6. loudness-normalise to a fixed target
    7. add a common dither floor
    8. write 16 kHz mono PCM_16 WAV

Step 4 is the codec equalisation. Decoding an MP3 to WAV does not undo
MP3; the quantisation artefacts survive. The only way to stop "has been
through MP3" from being a label is to put everything through it.

HONEST CAVEAT ON STEP 4

ElevenLabs clips end up with two MP3 generations while everything else
has one. That residual asymmetry is much weaker than MP3-vs-PCM -- the
44.1 kHz -> 16 kHz resample in step 2 already smears the original
MDCT frame grid before re-encoding -- but it is not zero. Re-run
probe_shortcuts.py afterwards and let it adjudicate rather than
assuming. If spectral features still separate the classes, the next
step is regenerating ElevenLabs as PCM, which this script cannot do.

ORDERING MATTERS

Loudness normalisation is last because the codec changes level. Silence
is trimmed both before and after the codec because the encoder adds
padding of its own. Doing these in a different order leaves a residue
of exactly the cue you were trying to remove.

WHAT ONE GAIN CANNOT FIX

A single gain per file can equalise exactly one level statistic. This
chain equalises loudness, so `rms_db` goes to chance -- but `peak_db`
and `crest_db` are then free to vary with the waveform's own peakiness,
and they may still separate the classes afterwards.

That residual is not purely an artefact. TTS output really is more
dynamically compressed than a field recording, and crest factor is a
property of the signal, not only of the file. Some of it is a cue a
detector is entitled to use. If crest-related features dominate the
second probe run, the honest move is to report it as a limitation
rather than to flatten the waveform until it disappears -- over-
normalising destroys the vocoder artefacts the detector needs.

Same for `noise_floor_db`: dither sets a common floor, but it cannot
manufacture room tone where there is none, only stop the floor from
being bit-exact silence.

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

    # preflight: is libmp3lame present?
    python training\normalise_corpus.py --check-ffmpeg

    # dry run (DEFAULT): plan + before/after on a sample, writes nothing
    python training\normalise_corpus.py --bonafide-root data\processed\bonafide --spoof-root meta-mms=data\processed\meta-mms --spoof-root elevenlabs=data\processed\elevenlabs --out-root data\normalised

    # do it
    python training\normalise_corpus.py --bonafide-root data\processed\bonafide --spoof-root meta-mms=data\processed\meta-mms --spoof-root elevenlabs=data\processed\elevenlabs --out-root data\normalised --apply --workers 4
"""

import argparse
import hashlib
import json
import shutil
import subprocess
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

# Conservative enough that post-normalisation clipping is rare. A
# limiter that only fires on some files would reintroduce exactly the
# level cue this is meant to remove.
DEFAULT_TARGET_DBFS = -26.0

# 64 kbps at 16 kHz mono is transparent enough not to damage the
# vocoder artefacts the detector actually needs, while still imposing
# a common codec history.
DEFAULT_MP3_BITRATE = "64k"

# Dither has to sit well ABOVE the PCM_16 LSB or quiet passages still
# round to bit-exact zero, which is the cue it was meant to remove.
# Measured fraction of silent samples quantising to zero:
#
#     -75 dBFS   5.8 LSB    6.80%      <- too quiet, leaks
#     -65 dBFS  18.4 LSB    2.14%
#     -60 dBFS  32.8 LSB    1.23%
#     -55 dBFS  58.3 LSB    0.68%      <- default
#     -50 dBFS 103.6 LSB    0.38%
#
# At -55 the noise sits 29 dB below the -26 dBFS speech target: audible
# as faint hiss, far gentler than the AMR-NB augmentation already in
# the methodology, and it also masks the unnaturally clean floor that
# synthesised audio has. Go louder if frac_exact_zero still leaks.
DEFAULT_DITHER_DBFS = -55.0


# ---------------------------------------------------------------------------
# FFMPEG
# ---------------------------------------------------------------------------


def ffmpeg_path():
    exe = shutil.which("ffmpeg")

    if exe is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. The codec equalisation step "
            "cannot run without it."
        )

    return exe


def check_ffmpeg():
    """Preflight. Returns 0 if the MP3 encoder is usable."""
    print("=" * 70)
    print("FFMPEG PREFLIGHT")
    print("=" * 70)

    try:
        exe = ffmpeg_path()
    except RuntimeError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    print(f"  ffmpeg: {exe}")

    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as e:
        print(f"  ERROR: could not list encoders: {e}", file=sys.stderr)
        return 1

    if "libmp3lame" not in out:
        print(
            "  ERROR: libmp3lame is not compiled into this ffmpeg.\n"
            "  Install a full build:\n"
            "      winget install Gyan.FFmpeg.Full\n"
            "  or:\n"
            "      conda install -c conda-forge ffmpeg",
            file=sys.stderr,
        )
        return 1

    print("  libmp3lame: present")

    # Prove the round trip actually works rather than trusting the
    # encoder list -- a codec can be listed and still fail at runtime.
    probe = (np.random.default_rng(0).normal(0, 0.1, TARGET_SR)).astype(
        np.float32
    )

    try:
        out_audio = mp3_roundtrip(probe, DEFAULT_MP3_BITRATE)
    except Exception as e:
        print(f"  ERROR: round trip failed: {e}", file=sys.stderr)
        return 1

    print(f"  round trip: {len(probe)} -> {len(out_audio)} samples, OK")
    print()
    print("RESULT: ready.")
    return 0


def _run_ffmpeg(args, payload):
    """Run ffmpeg with bytes on stdin, return bytes from stdout."""
    proc = subprocess.run(
        args,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )

    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1] if tail else "no stderr"
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {detail}")

    return proc.stdout


def mp3_roundtrip(x, bitrate=DEFAULT_MP3_BITRATE):
    """
    Encode to MP3 and decode straight back, entirely in memory.

    No temp files: the samples go in on stdin and come back on stdout,
    which matters when this runs across tens of thousands of clips.
    """
    exe = ffmpeg_path()

    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()

    encoded = _run_ffmpeg(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-f",
            "mp3",
            "pipe:1",
        ],
        pcm,
    )

    decoded = _run_ffmpeg(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            "pipe:1",
        ],
        encoded,
    )

    return np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768.0


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


def measure_lufs(x):
    """
    Integrated loudness per ITU-R BS.1770 if pyloudnorm is available.

    Returns None when it is not installed, so the caller can fall back
    to RMS. LUFS is preferable only because it is the citable standard;
    for killing the rms_db shortcut, RMS works just as well.
    """
    try:
        import pyloudnorm
    except ImportError:
        return None

    if len(x) < TARGET_SR // 2:
        return None

    try:
        meter = pyloudnorm.Meter(TARGET_SR)
        value = float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return None

    return value if np.isfinite(value) else None


def normalise_level(x, target_dbfs, mode):
    """
    Apply a single gain so every file lands at the same level.

    Returns (audio, applied_gain_db, clipped). `clipped` is reported
    rather than silently limited: a limiter firing on some files and
    not others would put a level cue straight back in.
    """
    used = "rms"

    if mode in ("auto", "lufs"):
        lufs = measure_lufs(x)

        if lufs is not None:
            gain_db = target_dbfs - lufs
            used = "lufs"
        elif mode == "lufs":
            raise RuntimeError(
                "pyloudnorm required for --loudness lufs "
                "(pip install pyloudnorm)"
            )
        else:
            gain_db = None
    else:
        gain_db = None

    if gain_db is None:
        rms = float(np.sqrt(np.mean(x**2) + 1e-20))
        gain_db = target_dbfs - 20.0 * np.log10(max(rms, 1e-10))

    y = x * (10.0 ** (gain_db / 20.0))

    peak = float(np.abs(y).max()) if y.size else 0.0
    clipped = peak > 0.999

    if clipped:
        y = y * (0.999 / peak)
        gain_db += 20.0 * np.log10(0.999 / peak)

    return y.astype(np.float32), float(gain_db), clipped, used


def add_dither(x, level_dbfs, seed_source):
    """
    Add a common low-level noise floor to every file.

    Two cues survive loudness normalisation and both come from the same
    place -- synthesised audio has no room tone:

        frac_exact_zero   a microphone chain essentially never produces
                          bit-exact zeros; a vocoder does
        noise_floor_db    the quietest frames of a recording sit on
                          room tone, a synthesiser's sit on nothing

    Adding identical broadband noise to everything sets a common floor,
    so neither is a label any more. The level is far below speech, so
    it does not touch the vocoder artefacts the detector needs.

    This is not a trick: dithering before quantising to 16-bit PCM is
    standard practice, and it is one sentence to justify in 4.3.2.

    Seeded from the source path, so a re-run reproduces the same output
    byte for byte.
    """
    if level_dbfs is None:
        return x

    # hashlib, not hash(): Python randomises string hashing per
    # process, so hash() would give a different noise realisation on
    # every run and the output would not be reproducible.
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

    amplitude = 10.0 ** (level_dbfs / 20.0)
    noise = rng.normal(0.0, amplitude, len(x)).astype(np.float32)

    return np.clip(x + noise, -0.999, 0.999).astype(np.float32)


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

        if job["equalise"]:
            x = mp3_roundtrip(x, job["bitrate"])
            x = trim_silence(x, job["top_db"])

        if len(x) < FRAME_LEN:
            result["error"] = "empty after trim"
            return result

        x, gain_db, clipped, used = normalise_level(
            x, job["target_dbfs"], job["loudness"]
        )
        result["loudness_backend"] = used

        if job["dither_dbfs"] is not None:
            x = add_dither(x, job["dither_dbfs"], job["src"])

        result["gain_db"] = gain_db
        result["clipped"] = bool(clipped)
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
                    "target_dbfs": args.target_dbfs,
                    "loudness": args.loudness,
                    "equalise": not args.no_equalise,
                    "bitrate": args.bitrate,
                    "dither_dbfs": args.dither_dbfs,
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
            "       above and whether one source is mostly clipping."
        )

    backends = {}
    for r in ok:
        key = r.get("loudness_backend", "?")
        backends[key] = backends.get(key, 0) + 1

    print()
    print(
        "  loudness measured by: "
        + ", ".join(f"{k} x{v}" for k, v in sorted(backends.items()))
    )

    if backends.get("rms"):
        print(
            "    NOTE: RMS averages over pauses, gated BS.1770 (LUFS)\n"
            "    does not. Where the classes differ in how much silence\n"
            "    they carry, RMS leaves a residual level difference that\n"
            "    shows up as peak_db / crest_db separability. Install\n"
            "    pyloudnorm and re-run if those features are still\n"
            "    leaking:\n"
            "        pip install pyloudnorm"
        )

    clipped = sum(1 for r in ok if r.get("clipped"))

    if clipped:
        pct = 100.0 * clipped / len(ok)
        print()
        print(f"  peak-limited after gain: {clipped} ({pct:.2f}%)")

        if pct > 2.0:
            print(
                "    Above ~2% this starts reintroducing a level cue.\n"
                "    Re-run with a lower --target-dbfs."
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
        "--target-dbfs",
        type=float,
        default=DEFAULT_TARGET_DBFS,
        help=f"level target (default: {DEFAULT_TARGET_DBFS})",
    )
    parser.add_argument(
        "--loudness",
        choices=["auto", "lufs", "rms"],
        default="auto",
        help=(
            "auto uses ITU-R BS.1770 when pyloudnorm is installed and "
            "falls back to RMS otherwise (default: auto)"
        ),
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=40.0,
        help="silence trim threshold below peak frame (default: 40)",
    )
    parser.add_argument(
        "--bitrate",
        default=DEFAULT_MP3_BITRATE,
        help=f"MP3 bitrate for equalisation (default: {DEFAULT_MP3_BITRATE})",
    )
    parser.add_argument(
        "--dither-dbfs",
        type=float,
        default=DEFAULT_DITHER_DBFS,
        help=(
            "common noise floor added to every file (default: "
            f"{DEFAULT_DITHER_DBFS:g}). "
            "Closes the frac_exact_zero and noise_floor_db cues that "
            "survive loudness normalisation. Use --no-dither to skip."
        ),
    )
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="skip the dither step",
    )
    parser.add_argument(
        "--no-equalise",
        action="store_true",
        help=(
            "skip the MP3 round trip. Only sensible if the probe says "
            "container/subtype are already balanced across classes."
        ),
    )

    parser.add_argument("--check-ffmpeg", action="store_true")
    parser.add_argument(
        "--report-json",
        default=None,
        help="write the per-file record to this JSON file",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.check_ffmpeg:
        return check_ffmpeg()

    if args.no_dither:
        args.dither_dbfs = None

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

    if not args.no_equalise:
        try:
            ffmpeg_path()
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(
                "Run --check-ffmpeg for details, or --no-equalise to "
                "skip the codec step.",
                file=sys.stderr,
            )
            return 1

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
    codec_step = "" if args.no_equalise else f"MP3 {args.bitrate} -> trim -> "
    dither_step = (
        ""
        if args.dither_dbfs is None
        else f" -> dither {args.dither_dbfs:g} dBFS"
    )
    print(
        f"  chain        : 16 kHz mono -> trim -> {codec_step}"
        f"{args.loudness} {args.target_dbfs:g} dBFS{dither_step}"
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
