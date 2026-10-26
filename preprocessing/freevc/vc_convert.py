r"""
vc_convert.py

Generates the voice-conversion spoof half from the pairing written by
make_vc_pairs.py.

This file was previously a script for testing three different VC models. The final study utilizes FreeVC.

For every bona fide clip of source speaker A, the audio is converted to
the voice of target speaker B, giving A's words in B's voice. Output
lands beside the other synthesis halves:

    data/processed/bonafide/<A>/<stem>.wav        ->  input
    data/processed/voice-conv/<A>/<stem>.1.wav    ->  output

USAGE

    *Dry Run*
    python preprocessing\freevc\vc_convert.py --model freevc24

    *Formal Generation*
    python preprocessing\freevc\vc_convert.py --model freevc24 --apply

    *Verification*
    python preprocessing\freevc\vc_convert.py --verify

    *Smoke Test*
    python preprocessing\freevc\vc_convert.py --model freevc24 --apply --limit 5

THE MAPPING IS RE-CHECKED BEFORE ANYTHING IS GENERATED

check_pairs() from make_vc_pairs.py runs again here against the CURRENT
splits. If a speaker has moved split since the mapping was written, this
refuses to generate. A cross-split pairing puts one voice on both sides
of the train/test line and nothing downstream can see it, because the
converted clip carries the SOURCE speaker's id.
"""

import argparse
import csv
import re
import random
import sys
import time
from pathlib import Path

# Project root, so `preprocessing.freevc.*` resolves when this file is run
# directly. parents[0]=freevc, parents[1]=preprocessing, parents[2]=root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Missing dependency(s). Run: pip install soundfile numpy",
          file=sys.stderr)
    sys.exit(1)

from preprocessing.freevc.make_vc_pairs import check_pairs, load_split_map, load_speakers

MODELS = {
    "freevc24": "voice_conversion_models/multilingual/vctk/freevc24",
}

REF_SECONDS = {"freevc24": 20.0}

SPOOF_SUFFIX = ".1"


def bonafide_stem(stem: str) -> str:
    """0200.xxx.1 -> 0200.xxx"""
    return re.sub(r"[.]\d$", "", stem)

# A converted clip that still correlates this strongly with its input is
# very likely a pass-through: the model loaded but did nothing.
NOOP_CORR = 0.50


def load_pairs(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: no pairing at {path}\n"
              f"       Run: python preprocessing\\freevc\\make_vc_pairs.py --apply",
              file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bona_clips(root: Path, spk: str) -> list:
    d = root / spk
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".wav")


def out_path(vc_root: Path, src: str, clip: Path) -> Path:
    return vc_root / src / f"{clip.stem}{SPOOF_SUFFIX}.wav"


def ref_arg(ref):
    """target_wav for the coqui API: a list stays a list."""
    return [str(p) for p in ref] if isinstance(ref, list) else str(ref)


def build_reference(tgt: str, bona_root: Path, seconds: float,
                    ref_dir: Path, seed: int, chunked: bool = False):
    """
    One reference wav per target speaker, concatenated from that
    speaker's own bona fide clips until the duration budget is met.

    Deterministic: random.Random seeded with a string uses sha512 of the
    string, so the same speaker draws the same clips on every machine
    and every rerun.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    dst = ref_dir / f"{tgt}.wav"

    clips = bona_clips(bona_root, tgt)
    if not clips:
        return None, []

    rng = random.Random(f"vcref-{tgt}-{seed}")
    order = clips[:]
    rng.shuffle(order)

    chunks, used, total, sr = [], [], 0.0, None
    for c in order:
        try:
            x, csr = sf.read(c, dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr is None:
            sr = csr
        elif csr != sr:
            continue
        chunks.append(x)
        used.append(c.name)
        total += len(x) / csr
        if total >= seconds:
            break

    if not chunks:
        return None, []

    if chunked:
        # Never concatenated, never written -- see CHUNKED_REFERENCE.
        return [c for c in order[:len(used)]], used

    if not dst.exists():
        sf.write(dst, np.concatenate(chunks), sr, subtype="PCM_16")
    return dst, used


def pearson(a, b) -> float:
    n = min(len(a), len(b))
    if n < 100:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / d) if d > 0 else 0.0


SR_PROBE = 16000


def _mel_filterbank(n_filt=26, nfft=512, sr=SR_PROBE):
    def h2m(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def m2h(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = m2h(np.linspace(h2m(0), h2m(sr / 2), n_filt + 2))
    b = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_filt, nfft // 2 + 1))
    for i in range(1, n_filt + 1):
        lo, mid, hi = b[i - 1], b[i], b[i + 1]
        for k in range(lo, mid):
            fb[i - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[i - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


_FB = _mel_filterbank()
_DCT = np.array([[np.cos(np.pi * i * (2 * j + 1) / 52) for j in range(26)]
                 for i in range(20)])


def _resample(x, sr_in, sr_out=SR_PROBE):
    """FFT resampling. No librosa, no scipy, identical on every machine."""
    if sr_in == sr_out:
        return x
    m = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    keep = m // 2 + 1
    X = X[:keep] if keep < len(X) else np.pad(X, (0, keep - len(X)))
    return (np.fft.irfft(X, m) * (m / len(x))).astype(np.float32)


def _mfcc_mean(path):
    """Mean MFCC c1..c19 of one clip, or None."""
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = _resample(x, sr)
    if len(x) < SR_PROBE // 4:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::160]
    if len(frames) < 10:
        return None
    p = np.abs(np.fft.rfft(frames * np.hamming(512), 512)) ** 2 / 512
    return (_DCT @ np.log(_FB @ p.T + 1e-10))[1:].mean(axis=1)


def mfcc_centroid(paths: list):
    r"""
    Mean MFCC over a set of clips, L2-normalised.

    A cheap stand-in for a speaker embedding. It is good enough to be
    worth running -- on this corpus it gets 16/16 rank-1 speaker
    identification on bona fide audio -- but it is NOT a
    speaker-similarity score for the paper. That wants a real ASV
    system (ECAPA-TDNN / x-vector) reported separately.
    """
    vecs = [m for m in (_mfcc_mean(p) for p in paths) if m is not None]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cosine(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return float(a @ b)


def centre(vecs: dict) -> dict:
    r"""
    Subtract the set mean, then renormalise.

    Every converted clip carries the SAME vocoder colouration, and in
    raw MFCC space that shared offset swamps the speaker. Measured on
    this corpus: converted clips from unrelated pairs resemble each
    other (0.9704) more than each resembles its own target (0.9491),
    and more than two real speakers resemble each other (0.9566).

    Centring each class on its own mean removes the offset. It moved
    the target's median rank from 25/70 to 10/70 (chance 35) and top-5
    from 14/70 to 29/70. Without it this probe reports the vocoder
    fingerprint and calls it a failed conversion.

    This is the same idea as mean-normalising embeddings before cosine
    scoring in speaker verification.
    """
    if not vecs:
        return {}
    mu = np.mean(list(vecs.values()), axis=0)
    out = {}
    for k, v in vecs.items():
        c = v - mu
        n = np.linalg.norm(c)
        out[k] = c / n if n > 0 else c
    return out


def median_pairwise(vecs: dict) -> float:
    ks = sorted(vecs)
    vals = [float(vecs[a] @ vecs[b])
            for i, a in enumerate(ks) for b in ks[i + 1:]]
    return float(np.median(vals)) if vals else float("nan")


def guard(pairs_rows: list, args) -> dict:
    """Re-run the pairing checks against the CURRENT splits."""
    pairs = {r["source_speaker"]: r["target_speaker"] for r in pairs_rows}
    recorded = {r["source_speaker"]: r.get("split") for r in pairs_rows}

    split_of = load_split_map(Path(args.splits_dir))
    vc_set = set(load_speakers(Path(args.vc_speakers)))
    bona_root = Path(args.bonafide_root)

    checks, problems = check_pairs(pairs, split_of, vc_set, bona_root)

    drifted = [(a, recorded[a], split_of[a]) for a in pairs
               if a in split_of and recorded.get(a)
               and recorded[a] != split_of[a]]
    checks.append(("recorded split still matches the manifests", not drifted))
    if drifted:
        problems.append(
            f"{len(drifted)} speaker(s) changed split since the mapping was "
            f"written: {drifted[:3]}")

    print("  mapping guard (re-checked against the current splits)")
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    print()

    if problems:
        print("REFUSING TO GENERATE:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\n  Fix the mapping first:\n"
              "    python preprocessing\\freevc\\make_vc_pairs.py --verify",
              file=sys.stderr)
        sys.exit(1)

    return split_of


def _patch_voice_cache_device(api):
    r"""
    coqui loads a cached voice with torch.load(map_location="cpu") and
    never moves it to the model's device (TTS/utils/voices.py). The
    FIRST call computes the embedding on the GPU and works; every later
    call reads it back onto the CPU and dies with

        Expected all tensors to be on the same device, but got weight is
        on cuda:0, different from other tensors on cpu

    Caching the voice is the whole point -- it is what stops the target
    embedding being recomputed for all 28,180 clips -- so the fix is to
    move the loaded tensors, not to abandon the cache.
    """
    import torch

    model = getattr(api.voice_converter, "vc_model", None)
    if model is None or not hasattr(model, "clone_voice"):
        return None

    original = model.clone_voice

    def on_device(*a, **kw):
        voice = original(*a, **kw)
        dev = next(model.parameters()).device
        return {k: (v.to(dev) if torch.is_tensor(v) else v)
                for k, v in voice.items()}

    model.clone_voice = on_device
    return "cached voices moved to the model device"


def _patch_torchaudio_load():
    r"""
    From torch 2.9, torchaudio.load routes through torchcodec, which
    needs FFmpeg SHARED libraries (libtorchcodec_coreN.dll on Windows).
    Without them knnvc cannot read a single file.

    Every clip in this corpus is PCM WAV, which soundfile reads without
    FFmpeg at all -- the same backend torchaudio itself used for years.
    Only installed if the real loader is actually broken, so this
    disappears by itself if FFmpeg is ever installed.
    """
    import tempfile

    import torch
    import torchaudio

    probe = Path(tempfile.gettempdir()) / "vc_convert_probe.wav"
    if not probe.exists():
        sf.write(probe, np.zeros(1600, dtype="float32"), 16000,
                 subtype="PCM_16")
    try:
        torchaudio.load(str(probe))
        return None
    except Exception:
        pass

    def load_via_soundfile(uri, *a, **kw):
        x, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        return torch.from_numpy(x.T.copy()), sr

    torchaudio.load = load_via_soundfile
    return ("torchaudio.load -> soundfile (torchcodec has no FFmpeg "
            "shared libs)")


def load_converter(model_key: str, device: str):
    import importlib.metadata as md

    try:
        from TTS.api import TTS
    except ImportError as e:
        if "torchcodec" not in str(e).lower():
            raise
        print(
            "\nERROR: coqui-tts refuses to import without torchcodec.\n"
            "\n"
            "  From torch 2.9 onward TTS/__init__.py checks that the package\n"
            "  exists -- importlib.util.find_spec, nothing more. FreeVC and\n"
            "  OpenVoice never call it; they decode with librosa. Satisfying\n"
            "  the check is enough.\n"
            "\n"
            "    venv\\Scripts\\python.exe -m pip install --no-deps torchcodec\n"
            "\n"
            "  --no-deps matters. torchcodec declares NO torch pin, so a plain\n"
            "  `pip install coqui-tts[codec]` re-resolves the whole coqui-tts\n"
            "  dependency tree inside the venv you train in -- which is how a\n"
            "  working torch/deepfense pair gets quietly replaced.\n"
            "\n"
            "  knnvc is the exception: it decodes via torchaudio, which now\n"
            "  routes through torchcodec and needs FFmpeg SHARED libraries on\n"
            "  PATH. If knnvc cannot read audio, use freevc24.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    env = []
    for pkg in ("torch", "torchcodec", "deepfense", "coqui-tts"):
        try:
            env.append(f"{pkg} {md.version(pkg)}")
        except Exception:
            env.append(f"{pkg} -")
    print("  env   " + "   ".join(env))
    print(f"  loading {model_key} ... (first run downloads the checkpoint)")
    api = TTS(MODELS[model_key]).to(device)
    sr = api.voice_converter.output_sample_rate

    for note in (_patch_voice_cache_device(api), _patch_torchaudio_load()):
        if note:
            print(f"  shim  {note}")

    print(f"  ready. native output rate {sr} Hz\n")
    return api, sr


def as_audio(wav) -> np.ndarray:
    r"""
    Normalise whatever a backend hands back into 1-D float32 numpy.

    The three models do not agree. FreeVC ends its voice_conversion with
    .data.cpu().float().numpy(); knnvc returns a torch tensor still on
    the GPU, which numpy refuses to touch:

        TypeError: can't convert cuda:0 device type tensor to numpy

    Funnelling every backend through here means the rest of the script
    never has to know which one is running.
    """
    if hasattr(wav, "detach"):
        wav = wav.detach().to("cpu")
    wav = np.asarray(wav, dtype=np.float32)
    return np.ravel(np.squeeze(wav))


def convert_one(api, src_wav: Path, tgt_id: str, voice_dir: Path):
    """Convert using the cached target voice. Returns 1-D float32 numpy."""
    return as_audio(api.voice_conversion(
        source_wav=str(src_wav),
        target_wav=None,
        speaker=tgt_id,
        voice_dir=str(voice_dir),
    ))


def do_audition(args, pairs_rows):
    out = Path(args.audition_dir)
    out.mkdir(parents=True, exist_ok=True)
    bona_root = Path(args.bonafide_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rng = random.Random(args.seed)
    picks = rng.sample(pairs_rows, min(args.audition_pairs, len(pairs_rows)))

    jobs = []
    for r in picks:
        src, tgt = r["source_speaker"], r["target_speaker"]
        clips = bona_clips(bona_root, src)[: args.audition_clips]
        for c in clips:
            jobs.append((src, tgt, c))

    print(f"  auditioning {len(jobs)} clip(s) x {len(models)} model(s)"
          f"  ->  {out}\n")

    notes = [
        "VC AUDITION",
        "",
        "Each group is one source clip. Listen in this order:",
        "",
        "  .ORIGINAL.wav    the real speaker saying it",
        "  .TARGET-REF.wav  the voice it is supposed to become",
        "  .<model>.wav     the conversion",
        "",
        "Three questions, in order of how much they matter:",
        "",
        "  1. Are the Cebuano words still intelligible and unchanged?",
        "     VC is text-free, so it cannot mispronounce -- but it can",
        "     smear. If the words survive, the content control holds.",
        "  2. Does it sound like TARGET-REF rather than ORIGINAL?",
        "     If it still sounds like the original speaker, the attack",
        "     is weak and the speaker confound is not fixed.",
        "  3. Does it sound like a recording of a person, or like an",
        "     artefact? Obvious buzz makes detection trivial again.",
        "",
    ]

    for mk in models:
        api, sr = load_converter(mk, args.device)
        vdir = Path(args.voice_cache) / mk
        built = set()
        for src, tgt, clip in jobs:
            if tgt not in built:
                ref, _ = build_reference(tgt, bona_root,
                                         REF_SECONDS.get(mk, 20.0),
                                         Path(args.ref_dir), args.seed,
                                         chunked=mk in CHUNKED_REFERENCE)
                if ref is None:
                    print(f"    skip {tgt}: no bona fide audio")
                    continue
                try:
                    api.voice_conversion(source_wav=str(clip),
                                         target_wav=ref_arg(ref),
                                         speaker=tgt, voice_dir=str(vdir))
                except Exception as e:
                    print(f"    FAIL building voice {tgt} [{mk}]: "
                          f"{type(e).__name__}: {e}")
                    continue
                built.add(tgt)

            tag = f"{src}-to-{tgt}.{clip.stem}"
            try:
                wav = convert_one(api, clip, tgt, vdir)
            except Exception as e:
                print(f"    FAIL {tag} [{mk}]: {type(e).__name__}: {e}")
                continue
            sf.write(out / f"{tag}.{mk}.wav", wav, sr, subtype="PCM_16")

            o, osr = sf.read(clip, dtype="float32")
            sf.write(out / f"{tag}.ORIGINAL.wav", o, osr, subtype="PCM_16")
            ref = (Path(args.ref_dir) / f"{tgt}.wav") if not isinstance(
                ref, list) else Path(ref[0])
            if ref.exists():
                rw, rsr = sf.read(ref, dtype="float32", frames=int(8 * 16000))
                sf.write(out / f"{tag}.TARGET-REF.wav", rw, rsr,
                         subtype="PCM_16")

            corr = pearson(o, wav)
            print(f"    {tag:44s} {mk:13s} {len(wav)/sr:5.2f}s  "
                  f"corr-with-input {corr:.3f}"
                  f"{'  <-- NO-OP?' if corr > NOOP_CORR else ''}")
            notes.append(f"{tag}  [{mk}]  {len(wav)/sr:.2f}s  "
                         f"corr-with-input {corr:.3f}")
        del api

    (out / "AUDITION.txt").write_text("\n".join(notes), encoding="utf-8")
    print(f"\n  wrote {out / 'AUDITION.txt'}")
    print("  Listen before committing to a full run.")
    return 0


def do_generate(args, pairs_rows, apply: bool):
    bona_root = Path(args.bonafide_root)
    vc_root = Path(args.out_root)
    ref_dir = Path(args.ref_dir)
    vdir = Path(args.voice_cache) / args.model

    plan, total, existing = [], 0, 0
    for r in pairs_rows:
        src, tgt = r["source_speaker"], r["target_speaker"]
        clips = bona_clips(bona_root, src)
        if args.limit:
            clips = clips[: args.limit]
        todo = [c for c in clips
                if args.overwrite or not out_path(vc_root, src, c).exists()]
        existing += len(clips) - len(todo)
        total += len(clips)
        plan.append((src, tgt, r.get("split", "?"), clips, todo))

    pending = sum(len(t) for _, _, _, _, t in plan)
    print(f"  model        {args.model}  ({MODELS[args.model]})")
    print(f"  source       {bona_root}")
    print(f"  destination  {vc_root}")
    print(f"  speakers     {len(plan)}")
    print(f"  clips        {total} total   {existing} already written   "
          f"{pending} to generate")
    print()

    if not apply:
        for src, tgt, sp, clips, todo in plan[:5]:
            print(f"    {src} -> {tgt}   {sp:5s}  {len(todo):4d} to do")
        if len(plan) > 5:
            print(f"    ... and {len(plan) - 5} more")
        print("\n  Dry run. Add --apply to generate.")
        return 0

    if pending == 0:
        print("  Nothing to do. Everything is already written.")
        print("  Run --verify to check it.")
        return 0

    api, sr = load_converter(args.model, args.device)
    ref_rows, report = [], []
    done, failed, t0 = 0, 0, time.time()

    for src, tgt, sp, clips, todo in plan:
        if not todo:
            continue
        ref, used = build_reference(tgt, bona_root,
                                    REF_SECONDS.get(args.model, 20.0),
                                    ref_dir, args.seed,
                                    chunked=args.model in CHUNKED_REFERENCE)
        if ref is None:
            print(f"  SKIP {src} -> {tgt}: target has no bona fide audio")
            failed += len(todo)
            continue
        for name in used:
            ref_rows.append({"target_speaker": tgt, "clip": name})

        # Compute the target voice once, then reuse it from the cache.
        try:
            api.voice_conversion(source_wav=str(todo[0]),
                                 target_wav=ref_arg(ref),
                                 speaker=tgt, voice_dir=str(vdir))
        except Exception as e:
            print(f"  SKIP {src} -> {tgt}: could not build the target "
                  f"voice: {type(e).__name__}: {e}")
            failed += len(todo)
            continue

        (vc_root / src).mkdir(parents=True, exist_ok=True)
        sdone, sfail, st = 0, 0, time.time()
        for clip in todo:
            dst = out_path(vc_root, src, clip)
            try:
                wav = convert_one(api, clip, tgt, vdir)
                if wav.size < sr // 10:
                    raise RuntimeError(f"output too short ({wav.size} samples)")
                sf.write(dst, wav, sr, subtype="PCM_16")
                sdone += 1
            except Exception as e:
                sfail += 1
                if sfail <= 3:
                    print(f"    FAIL {clip.name}: {type(e).__name__}: {e}")
                if dst.exists() and dst.stat().st_size == 0:
                    dst.unlink()

        done += sdone
        failed += sfail
        el = time.time() - st
        rate = done / max(time.time() - t0, 1e-6)
        eta = (pending - done) / rate / 60 if rate > 0 else 0
        print(f"  {src} -> {tgt}  {sp:5s}  {sdone:4d} ok  {sfail:3d} fail  "
              f"{el:6.1f}s   [{done}/{pending}]  eta {eta:5.1f} min")
        report.append({"source_speaker": src, "target_speaker": tgt,
                       "split": sp, "written": sdone, "failed": sfail,
                       "seconds": round(el, 1)})

    mins = (time.time() - t0) / 60
    print(f"\n  {done} written, {failed} failed, {mins:.1f} min")

    _write_csv(Path(args.ref_manifest), ref_rows,
               ["target_speaker", "clip"])
    _write_csv(Path(args.report), report,
               ["source_speaker", "target_speaker", "split", "written",
                "failed", "seconds"])
    print(f"  reference clips -> {args.ref_manifest}")
    print(f"  per-speaker log -> {args.report}")

    if failed:
        print("\n  Some clips failed. Rerun the same command -- it resumes.")
        return 1

    print("\n  Next:")
    print("    python preprocessing\\freevc\\vc_convert.py --verify")
    print("    python training\\normalise_corpus.py "
          "--bonafide-root data\\processed\\bonafide "
          "--spoof-root voice-conv=data\\processed\\voice-conv "
          "--spoof-root elevenlabs=data\\processed\\elevenlabs "
          "--out-root data\\normalised --apply --workers 6")
    return 0


def _write_csv(path: Path, rows: list, fields: list):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def do_verify(args, pairs_rows):
    bona_root = Path(args.bonafide_root)
    vc_root = Path(args.out_root)

    if not vc_root.is_dir():
        print(f"ERROR: nothing at {vc_root}", file=sys.stderr)
        return 1

    # ---- completeness ------------------------------------------------
    missing, empty, expected, found = [], [], 0, 0
    have = {}
    for r in pairs_rows:
        src = r["source_speaker"]
        got = []
        for c in bona_clips(bona_root, src):
            expected += 1
            d = out_path(vc_root, src, c)
            if not d.exists():
                missing.append(str(d))
            elif d.stat().st_size < 1024:
                empty.append(str(d))
            else:
                found += 1
                got.append(d)
        if got:
            have[src] = got

    partial = bool(missing) and found > 0
    print(f"  expected {expected} clips, found {found}"
          f"   ({len(have)} speakers)")
    print(f"  missing {len(missing)}, truncated {len(empty)}")
    if partial:
        print("  -> PARTIAL run. The speaker probe still works on what "
              "exists;\n     completeness is reported separately below.")
    print()

    # ---- duration + pass-through -------------------------------------
    rng = random.Random(args.seed)
    corrs, ratios = [], []
    for src, outs in list(have.items()):
        for d in outs[: args.verify_clips]:
            c = bona_root / src / f"{bonafide_stem(d.stem)}.wav"
            if not c.exists():
                continue
            a, asr = sf.read(c, dtype="float32")
            b, bsr = sf.read(d, dtype="float32")
            ratios.append((len(b) / bsr) / max(len(a) / asr, 1e-9))
            corrs.append(pearson(a, _resample(b, bsr, asr)))

    med_corr = float(np.median(corrs)) if corrs else float("nan")
    med_rat = float(np.median(ratios)) if ratios else float("nan")

    # ---- speaker probe: rank, not threshold --------------------------
    print(f"  building speaker probe "
          f"({args.probe_clips} converted / {args.gallery_clips} bona fide "
          f"clips per speaker) ...")

    conv = {}
    for src, outs in have.items():
        picks = outs[:] 
        rng.shuffle(picks)
        v = mfcc_centroid(picks[: args.probe_clips])
        if v is not None:
            conv[src] = v

    gallery = {}
    for r in pairs_rows:
        for s in (r["source_speaker"], r["target_speaker"]):
            if s not in gallery:
                v = mfcc_centroid(
                    bona_clips(bona_root, s)[: args.gallery_clips])
                if v is not None:
                    gallery[s] = v

    cohesion = median_pairwise(conv)
    natural = median_pairwise(gallery)

    cc, gg = centre(conv), centre(gallery)
    ids = sorted(gg)
    M = np.stack([gg[i] for i in ids]) if ids else None

    rt, rs = [], []
    for r in pairs_rows:
        s, t = r["source_speaker"], r["target_speaker"]
        if s not in cc or t not in gg or s not in gg or M is None:
            continue
        order = [ids[i] for i in np.argsort(-(M @ cc[s]))]
        rt.append(order.index(t) + 1)
        rs.append(order.index(s) + 1)

    n_gal = len(ids)
    rt, rs = np.array(rt), np.array(rs)
    chance = (n_gal + 1) / 2
    med_t = float(np.median(rt)) if len(rt) else float("nan")
    med_s = float(np.median(rs)) if len(rs) else float("nan")
    top5 = int((rt <= 5).sum()) if len(rt) else 0

    print()
    print(f"  median duration ratio        {med_rat:.3f}")
    print(f"  median corr with input       {med_corr:.3f}")
    print()
    print(f"  vocoder cohesion   converted-to-converted  {cohesion:.4f}")
    print(f"                     bonafide-to-bonafide    {natural:.4f}")
    if cohesion > natural:
        print(f"                     excess {cohesion - natural:+.4f}  "
              "-- a shared synthesis colouration, which is")
        print("                     exactly the artefact the detector "
              "should be finding.")
    print()
    print(f"  speaker probe over {n_gal} speakers "
          f"(rank 1 = best, chance = {chance:.1f})")
    print(f"    median rank of the TARGET voice   {med_t:5.1f}")
    print(f"    median rank of the SOURCE voice   {med_s:5.1f}")
    print(f"    target in top 5                   {top5}/{len(rt)}   "
          f"(chance {5 * len(rt) / max(n_gal, 1):.0f})")
    print()

    checks = [
        ("every expected clip written", not missing),
        ("no truncated files", not empty),
        ("duration preserved (median ratio 0.8-1.25)",
         0.8 <= med_rat <= 1.25),
        (f"not a pass-through (median corr {med_corr:.3f} < {NOOP_CORR})",
         med_corr < NOOP_CORR),
        ("target voice ranks better than chance",
         len(rt) > 0 and med_t < chance * 0.75),
        ("target voice ranks above the source voice",
         len(rt) > 0 and med_t < med_s),
    ]
    if partial and args.allow_partial:
        checks[0] = ("every expected clip written (SKIPPED: partial run)",
                     True)

    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    print()

    if missing[:3]:
        print("  first missing:", *missing[:3], sep="\n    ")
        print()

    bad = [c for c, ok in checks if not ok]
    if bad:
        print("  NOT ready to normalise. Failed: " + "; ".join(bad))
        if partial and not args.allow_partial:
            print("\n  If this was deliberately a --limit run, rerun with "
                  "--allow-partial\n  to check everything except "
                  "completeness.")
        return 1

    print("  The VC half is consistent with the pairing.")
    print("  The MFCC probe is a build guard, not a paper metric -- report")
    print("  speaker similarity from a real ASV system separately.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate the voice-conversion spoof half.")
    ap.add_argument("--pairs", default="manifests/vc_speaker_pairs.csv")
    ap.add_argument("--bonafide-root", default="data/processed/bonafide")
    ap.add_argument("--out-root", default="data/processed/voice-conv")
    ap.add_argument("--ref-dir", default="data/vc_references")
    ap.add_argument("--voice-cache", default="data/vc_voice_cache")
    ap.add_argument("--ref-manifest",
                    default="manifests/vc_reference_clips.csv")
    ap.add_argument("--report", default="manifests/vc_generation_report.csv")
    ap.add_argument("--splits-dir", default="manifests/splits_norm")
    ap.add_argument("--vc-speakers",
                    default="manifests/vc_selected_speakers.csv")
    ap.add_argument("--model", default="freevc24", choices=sorted(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "val", "test"],
                    help="only convert speakers in this split")
    ap.add_argument("--limit", type=int, default=0,
                    help="clips per speaker, for a smoke test (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--audition", action="store_true")
    ap.add_argument("--audition-dir", default="data/audition/vc")
    ap.add_argument("--audition-pairs", type=int, default=3)
    ap.add_argument("--audition-clips", type=int, default=2)
    ap.add_argument("--models", default="freevc24,knnvc,openvoice_v2",
                    help="audition only: comma-separated")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-clips", type=int, default=10)
    ap.add_argument("--probe-clips", type=int, default=25)
    ap.add_argument("--gallery-clips", type=int, default=15)
    ap.add_argument("--allow-partial", action="store_true",
                    help="skip the completeness check")
    args = ap.parse_args()

    print()
    pairs_rows = load_pairs(Path(args.pairs))
    print(f"  pairing  {args.pairs}   ({len(pairs_rows)} pairs)\n")

    if args.verify:
        return do_verify(args, pairs_rows)

    guard(pairs_rows, args)

    # The guard runs on the FULL mapping -- the derangement and
    # split checks are only meaningful over all 70 pairs. Filtering
    # happens after, so a subset run still proves the whole mapping.
    if args.split:
        pairs_rows = [r for r in pairs_rows if r.get("split") == args.split]
        print(f"  filtered to split '{args.split}': "
              f"{len(pairs_rows)} pair(s)\n")
        if not pairs_rows:
            print(f"ERROR: no pairs in split '{args.split}'", file=sys.stderr)
            return 1

    if args.audition:
        return do_audition(args, pairs_rows)
    return do_generate(args, pairs_rows, args.apply)


if __name__ == "__main__":
    sys.exit(main())
