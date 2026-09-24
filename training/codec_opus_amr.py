r"""
codec_opus_amr.py

On-the-fly Opus and AMR-NB codec augmentation, implementing the
bitrate policy in Section 4.3.2 of the thesis.

Why this exists:

    DeepFense ships a built-in "codec" augmentation, but it exposes
    only noise_ratio -- there is no way to select Opus vs AMR or to
    control bitrate. The methodology calls for:

        Opus     stochastic bitrate, 6 kbps - 32 kbps
        AMR-NB   cycling the eight standard source bitrates

    so the compression is implemented here with ffmpeg instead.

Nothing is written to disk. Audio is piped through ffmpeg in memory,
so no additional training files are generated -- matching the
"applied dynamically" requirement.

IMPORTANT -- AMR-NB is an 8 kHz codec:

    AMR-NB cannot encode 16 kHz audio. The transform therefore
    resamples 16k -> 8k, encodes, decodes, and resamples back to
    16 kHz. That downsample-upsample round trip is itself part of the
    channel degradation being simulated (it is what actually happens
    on a 2G/3G voice call), so it is intentional, not a bug.

    Opus encodes 16 kHz directly, no resampling needed.

IMPORTANT -- throughput, measured:

    Each clip costs TWO ffmpeg process spawns (encode, then decode),
    and process startup dominates the actual compression work.

    Benchmarked at ~368 ms per 10 s clip, single process
    (~2.7 clips/s). For a 62,433-clip epoch that is roughly:

        1 worker    ~6.4 hours of augmentation per epoch
        8 workers   ~48 minutes per epoch

    At 50 epochs this is the difference between a two-day run and a
    two-week one, so treat it as a real constraint, not a detail.

    Re-run --benchmark on YOUR machine before committing to a long
    run; core count changes this a lot.

    Mitigations, cheapest first:

      1. noise_ratio 0.5 (the config default) halves the cost -- only
         half the batch takes the ffmpeg path, and the model still
         sees both clean and degraded audio.
      2. Raise dataloader num_workers to at least 8.
      3. If it is still the bottleneck, replace this module with
         torchaudio's in-process codec support
         (torchaudio.io.AudioEffector), which avoids subprocess spawn
         entirely. Same augmentation, no process overhead -- but
         verify its AMR-NB support separately, since that is exactly
         the piece most builds omit.

Registering with DeepFense:

    Both transforms register themselves via @register_transform from
    deepfense.utils.registry, under the names:

        opus_codec
        amrnb_codec

    The decorator only fires when THIS MODULE IS IMPORTED, and the
    deepfense CLI does not import it. Launch training through
    training/run_training.py, which imports this module first:

        python training\run_training.py --config training\configs\train_main.yaml

    Running `deepfense train` directly on a config that references
    these will fail with a KeyError from the Transform registry. That
    is the correct failure -- loud, not a silent skip.

    Note on YAML argument shape: deepfense's build_transforms_pipeline
    pops "type" and passes every remaining key as a kwarg, so args are
    FLAT in the config, not nested under an `args:` block.

    Both are also usable directly, outside DeepFense, via
    apply_opus() / apply_amrnb().

Requires:

    ffmpeg on PATH, built with libopus and AMR-NB support.
    Verify with:  ffmpeg -codecs | findstr "opus amr"

    pip install numpy soundfile

Usage (standalone self-test -- run this BEFORE wiring into training):

    python training\codec_opus_amr.py --self-test

    python training\codec_opus_amr.py \
        --input data\processed\bonafide\0201\some.wav \
        --out-dir outputs\codec_preview

    python training\codec_opus_amr.py --benchmark --iterations 50
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print(
        "Missing dependency. Run: pip install numpy",
        file=sys.stderr,
    )
    sys.exit(1)

# Registration is optional so the module stays runnable standalone
# (--check-ffmpeg, --self-test, --benchmark) without deepfense present.
try:
    from deepfense.utils.registry import register_transform

    DEEPFENSE_AVAILABLE = True
except ImportError:

    def register_transform(name):
        def decorator(obj):
            return obj

        return decorator

    DEEPFENSE_AVAILABLE = False


TARGET_SR = 16000

# Opus: continuous range, sampled per clip (thesis: 6-32 kbps)
OPUS_BITRATE_MIN = 6000
OPUS_BITRATE_MAX = 32000

# AMR-NB: the eight standard source bitrates, in bits per second
AMRNB_BITRATES = (
    4750,
    5150,
    5900,
    6700,
    7400,
    7950,
    10200,
    12200,
)

AMRNB_SR = 8000

FFMPEG_TIMEOUT = 30


# ---------------------------------------------------------------------------
# FFMPEG
# ---------------------------------------------------------------------------


OPUS_ENCODER = "libopus"
AMRNB_ENCODER = "libopencore_amrnb"


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")

    if path is None:
        print(
            "ERROR: ffmpeg not found on PATH.\n"
            "Install with: winget install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)

    return path


def available_encoders() -> set:
    """Encoder names this ffmpeg build can actually ENCODE with."""
    require_ffmpeg()

    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=FFMPEG_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()

    names = set()

    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split()

        # encoder lines look like: " A....D libopus  Opus ..."
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])

    return names


def has_encoder(name: str) -> bool:
    return name in available_encoders()


def require_encoder(name: str, codec_label: str):
    """
    Fail loudly at construction rather than silently passing audio
    through unchanged.

    A missing encoder that degrades to a no-op is the dangerous case:
    training would appear to use codec augmentation while actually
    using none, and the ablation would be measuring nothing.
    """
    if has_encoder(name):
        return

    raise RuntimeError(
        f"{codec_label} augmentation requires the '{name}' encoder, "
        "which this ffmpeg build does not have.\n"
        "Run `python training/codec_opus_amr.py --check-ffmpeg` "
        "for details and remedies."
    )


def check_ffmpeg_report() -> int:
    """Preflight: report exactly which codecs are usable."""
    binary = require_ffmpeg()

    encoders = available_encoders()

    print("=" * 70)
    print("FFMPEG PREFLIGHT")
    print("=" * 70)
    print(f"Binary: {binary}")
    print()

    results = []

    for label, encoder in (
        ("Opus", OPUS_ENCODER),
        ("AMR-NB", AMRNB_ENCODER),
    ):
        ok = encoder in encoders
        results.append(ok)
        status = "AVAILABLE" if ok else "MISSING"
        print(f"  {label:8s} encoder '{encoder}': {status}")

    print()

    if all(results):
        print("RESULT: both codecs usable.")
        return 0

    print("RESULT: at least one codec is NOT usable.")
    print()
    print(
        "Note that DECODE support is not enough -- many ffmpeg builds\n"
        "ship AMR-NB as decode-only. Installing libopencore-amrnb as a\n"
        "system package does NOT help: ffmpeg must be COMPILED with\n"
        "--enable-libopencore-amrnb.\n"
    )
    print("Remedies:")
    print(
        "  Windows : install a FULL build (gyan.dev 'full' or BtbN),\n"
        "            not the 'essentials' build.\n"
        "            winget install Gyan.FFmpeg.Full\n"
        "  Conda   : conda install -c conda-forge ffmpeg\n"
        "  Check   : ffmpeg -encoders | findstr amr"
    )
    print()
    print(
        "Until AMR-NB encoding works, do NOT run the AMR arm of the\n"
        "experiment matrix -- the transform will raise rather than\n"
        "silently train without augmentation."
    )

    return 1


def _to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> signed 16-bit little-endian bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _from_pcm16(raw: bytes) -> np.ndarray:
    """signed 16-bit little-endian bytes -> float32 [-1, 1]."""
    if not raw:
        return np.zeros(0, dtype=np.float32)

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def _run_ffmpeg(args: list, payload: bytes) -> bytes:
    """Pipes raw PCM through ffmpeg and returns raw PCM back."""
    try:
        proc = subprocess.run(
            args,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print("WARNING: ffmpeg timed out", file=sys.stderr)
        return b""

    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace")
        print(
            f"WARNING: ffmpeg failed: {message.strip()[-300:]}",
            file=sys.stderr,
        )
        return b""

    return proc.stdout


def _match_length(
    processed: np.ndarray,
    original_length: int,
) -> np.ndarray:
    """
    Codecs pad or trim by a few samples. Force the original length so
    the transform is shape-preserving for the collator.
    """
    if processed.size == 0:
        return np.zeros(original_length, dtype=np.float32)

    if len(processed) >= original_length:
        return processed[:original_length]

    out = np.zeros(original_length, dtype=np.float32)
    out[: len(processed)] = processed
    return out


# ---------------------------------------------------------------------------
# CODECS
# ---------------------------------------------------------------------------


def apply_opus(
    audio: np.ndarray,
    bitrate: int,
    sample_rate: int = TARGET_SR,
) -> np.ndarray:
    """
    Encode to Opus at `bitrate` and decode back, in memory.

    Opus handles 16 kHz natively, so there is no resampling step.
    """
    original_length = len(audio)

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        str(bitrate),
        "-f",
        "ogg",
        "pipe:1",
    ]

    encoded = _run_ffmpeg(args, _to_pcm16(audio))

    if not encoded:
        return audio

    decode_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "ogg",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]

    decoded = _run_ffmpeg(decode_args, encoded)

    return _match_length(_from_pcm16(decoded), original_length)


def apply_amrnb(
    audio: np.ndarray,
    bitrate: int,
    sample_rate: int = TARGET_SR,
) -> np.ndarray:
    """
    Encode to AMR-NB at `bitrate` and decode back, in memory.

    AMR-NB is 8 kHz only, so this does 16k -> 8k -> encode -> decode
    -> 16k. The bandwidth loss from that round trip is part of the
    cellular channel being simulated.
    """
    original_length = len(audio)

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-ar",
        str(AMRNB_SR),  # AMR-NB requires 8 kHz
        "-ac",
        "1",
        "-c:a",
        "libopencore_amrnb",
        "-b:a",
        str(bitrate),
        "-f",
        "amr",
        "pipe:1",
    ]

    encoded = _run_ffmpeg(args, _to_pcm16(audio))

    if not encoded:
        return audio

    decode_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "amr",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),  # back up to 16 kHz
        "-ac",
        "1",
        "pipe:1",
    ]

    decoded = _run_ffmpeg(decode_args, encoded)

    return _match_length(_from_pcm16(decoded), original_length)


# ---------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------


@register_transform("opus_codec")
class OpusCodec:
    """
    YAML (args are FLAT -- no `args:` block):

        - type: "opus_codec"
          noise_ratio: 0.5
          bitrate_min: 6000
          bitrate_max: 32000

    Bitrate is drawn uniformly per clip, so the model sees the whole
    quality range rather than one fixed operating point.
    """

    def __init__(
        self,
        noise_ratio: float = 0.5,
        bitrate_min: int = OPUS_BITRATE_MIN,
        bitrate_max: int = OPUS_BITRATE_MAX,
        sample_rate: int = TARGET_SR,
        seed: int | None = None,
    ):
        require_encoder(OPUS_ENCODER, "Opus")

        self.noise_ratio = noise_ratio
        self.bitrate_min = bitrate_min
        self.bitrate_max = bitrate_max
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def __call__(self, audio: np.ndarray, *args, **kwargs):
        if self.rng.random() >= self.noise_ratio:
            return audio

        bitrate = self.rng.randint(self.bitrate_min, self.bitrate_max)

        return apply_opus(audio, bitrate, self.sample_rate)


@register_transform("amrnb_codec")
class AmrNbCodec:
    """
    YAML (args are FLAT -- no `args:` block):

        - type: "amrnb_codec"
          noise_ratio: 0.5
          bitrates: [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]

    AMR-NB has eight discrete source rates rather than a continuous
    range, so one is chosen per clip.
    """

    def __init__(
        self,
        noise_ratio: float = 0.5,
        bitrates: tuple = AMRNB_BITRATES,
        sample_rate: int = TARGET_SR,
        seed: int | None = None,
    ):
        require_encoder(AMRNB_ENCODER, "AMR-NB")

        self.noise_ratio = noise_ratio
        self.bitrates = tuple(bitrates)
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def __call__(self, audio: np.ndarray, *args, **kwargs):
        if self.rng.random() >= self.noise_ratio:
            return audio

        bitrate = self.rng.choice(self.bitrates)

        return apply_amrnb(audio, bitrate, self.sample_rate)


# ---------------------------------------------------------------------------
# SELF-TEST / BENCHMARK
# ---------------------------------------------------------------------------


def _tone(seconds: float = 10.0, sr: int = TARGET_SR) -> np.ndarray:
    """A speech-band sweep -- enough to hear codec damage."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sweep = np.sin(2 * np.pi * (200 + 600 * t / seconds) * t)
    return (0.4 * sweep).astype(np.float32)


def self_test():
    require_ffmpeg()

    print("=" * 70)
    print("CODEC SELF-TEST")
    print("=" * 70)

    audio = _tone()
    print(f"Input: {len(audio)} samples @ {TARGET_SR} Hz\n")

    ok = True

    for label, fn, rates in (
        ("Opus", apply_opus, (6000, 16000, 32000)),
        ("AMR-NB", apply_amrnb, AMRNB_BITRATES),
    ):
        print(f"{label}:")

        for bitrate in rates:
            out = fn(audio, bitrate)

            changed = not np.allclose(out, audio, atol=1e-6)
            same_len = len(out) == len(audio)
            rms = float(np.sqrt(np.mean(out**2)))

            status = "OK" if (changed and same_len) else "FAILED"

            if status == "FAILED":
                ok = False

            print(
                f"  {bitrate:6d} bps -> len={len(out)} rms={rms:.4f}  {status}"
            )

        print()

    print("RESULT:", "codecs working" if ok else "SOMETHING IS WRONG")
    print()
    print(
        "If AMR-NB failed, your ffmpeg lacks libopencore_amrnb.\n"
        "Check with: ffmpeg -codecs | findstr amr"
    )

    return 0 if ok else 1


def benchmark(iterations: int):
    require_ffmpeg()

    audio = _tone()

    print("=" * 70)
    print(f"THROUGHPUT BENCHMARK ({iterations} clips of 10 s)")
    print("=" * 70)

    for label, transform in (
        ("Opus", OpusCodec(noise_ratio=1.0, seed=0)),
        ("AMR-NB", AmrNbCodec(noise_ratio=1.0, seed=0)),
    ):
        start = time.perf_counter()

        for _ in range(iterations):
            transform(audio)

        elapsed = time.perf_counter() - start
        per_clip = elapsed / iterations

        print(
            f"{label:8s}: {elapsed:6.2f}s total, "
            f"{per_clip * 1000:7.1f} ms/clip, "
            f"{1 / per_clip:6.1f} clips/s (single process)"
        )

    print()
    print(
        "Compare against your GPU step time. If augmentation is "
        "slower than the forward/backward pass, the dataloader is "
        "your bottleneck -- raise num_workers or lower noise_ratio."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Opus / AMR-NB codec augmentation for DeepFense, plus "
            "standalone self-test and throughput benchmark."
        )
    )

    parser.add_argument(
        "--check-ffmpeg",
        action="store_true",
        help=(
            "Preflight: report whether this ffmpeg build can ENCODE "
            "Opus and AMR-NB. Run this first."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify ffmpeg can encode/decode both codecs",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure clips/second for each codec",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Clips to process in the benchmark (default: 25)",
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optional WAV to run through every bitrate",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/codec_preview",
        help="Where --input results are written",
    )

    args = parser.parse_args()

    if args.check_ffmpeg:
        sys.exit(check_ffmpeg_report())

    if args.self_test:
        sys.exit(self_test())

    if args.benchmark:
        benchmark(args.iterations)
        return

    if args.input:
        try:
            import soundfile as sf
        except ImportError:
            print(
                "Missing dependency. Run: pip install soundfile",
                file=sys.stderr,
            )
            sys.exit(1)

        require_ffmpeg()

        src = Path(args.input)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(src), dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        print(f"Loaded {src.name}: {len(audio)} samples @ {sr} Hz")

        for bitrate in (6000, 16000, 32000):
            out = apply_opus(audio, bitrate, sr)
            path = out_dir / f"{src.stem}.opus{bitrate // 1000}k.wav"
            sf.write(str(path), out, sr)
            print(f"  wrote {path}")

        for bitrate in AMRNB_BITRATES:
            out = apply_amrnb(audio, bitrate, sr)
            path = out_dir / f"{src.stem}.amr{bitrate}.wav"
            sf.write(str(path), out, sr)
            print(f"  wrote {path}")

        print(f"\nListen to these before trusting the augmentation.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
