r"""
audition_synthesis.py

Runs the SAME clips through every candidate synthesis backend and
writes the results side by side, so you can A/B them once and pick on
evidence instead of speculation.

WHY THIS EXISTS

Meta MMS gives one synthetic voice for Cebuano. That single voice is
present in train, val and test, and a detector trained on MMS alone
reaches ZERO errors on held-out speakers -- the task collapsed to voice
recognition. Per-source EER: meta-mms 0.000237, elevenlabs 0.012254.
A 52x gap.

No open multi-voice TTS supports Cebuano. Piper has no Philippine
language at all; XTTS has 17 languages and none of them is Cebuano;
MMS is architecturally one voice per language. So the replacement has
to come from somewhere else, and every option carries a different risk:

    vocoder     copy-synthesis. Takes AUDIO, not text, so language is
                irrelevant and it CANNOT fail for Cebuano reasons.
                Keeps the original speaker's voice, so no confound is
                possible. Classified as TTS-artefact data by the field
                (Wang & Yamagishi ICASSP 2023; WaveFake is built this
                way). Risk: none linguistically, but it is resynthesis
                rather than text-to-speech.

    omnivoice   lists Cebuano (12.17 h of training data). Genuine TTS.
                Their own docs warn of "unstable results for some
                low-resource languages" -- and 12 h is thin.

    xtts        no Cebuano. But it uses a SHARED BPE tokenizer across
                all languages rather than per-language text processing,
                and Cebuano orthography is Spanish-derived (five
                vowels, Spanish consonant conventions, heavy loanword
                influence). Driving it with language="es" on Cebuano
                text is a reasonable bet that nobody has published.

WHAT TO LISTEN FOR

Read AUDITION.txt as you go -- it holds the transcript for each clip,
so you can check the words actually came out.

    words       the sentence must be intelligible and correct. This is
                what kills xtts/omnivoice if they fail.
    voice       should resemble the .ORIGINAL.wav speaker.
    texture     a slightly synthetic quality is GOOD. That is the
                artefact your detector needs to find.

Whatever passes your ears goes to Dr. Goron for the linguistic
validation your section 4.10 already commits to -- he is checking for
exactly this: "mispronunciations, incorrect intonation, or unintended
drift into other Visayan dialects".

USAGE

    python preprocessing\audition_synthesis.py --clips 10

    python preprocessing\audition_synthesis.py --clips 10 ^
        --backends vocoder,xtts --device cuda

INSTALLS (each backend is optional; missing ones are skipped loudly)

    vocoder     already available -- uses transformers
    xtts        pip install coqui-tts
    omnivoice   pip install omnivoice
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Missing dependency. Run: pip install soundfile numpy",
          file=sys.stderr)
    sys.exit(1)

TARGET_SR = 16000
REF_MIN_S = 4.0     # zero-shot models want 3-10 s of reference
REF_MAX_S = 10.0


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def resample(wav, sr_from, sr_to=TARGET_SR):
    """Everything is compared at 16 kHz, the corpus rate."""
    if sr_from == sr_to:
        return wav
    try:
        import librosa
        return librosa.resample(wav, orig_sr=sr_from, target_sr=sr_to)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_from, sr_to)
        return resample_poly(wav, sr_to // g, sr_from // g)


def load_transcripts(root: Path) -> dict:
    """{clip_stem: text} from the per-speaker tab-separated files."""
    out = {}
    if not root.is_dir():
        return out
    for f in root.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            name, text = line.split("\t", 1)
            stem = name[:-4] if name.endswith(".wav") else name
            out[stem] = text.strip()
    return out


def build_reference(clips, exclude: Path, texts: dict):
    """
    Concatenate clips from one speaker into a 4-10 s reference.

    Zero-shot models clone from this. The target clip is excluded so
    the model is not simply handed the answer.
    """
    parts, words, total = [], [], 0.0
    for p in clips:
        if p == exclude:
            continue
        # A reference is useless to a zero-shot model without its
        # transcript, so only take clips whose text we have.
        txt = texts.get(p.stem)
        if not txt:
            continue
        wav, sr = sf.read(p, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = resample(wav, sr)
        parts.append(wav)
        words.append(txt)
        total += len(wav) / TARGET_SR
        if total >= REF_MIN_S:
            break
    if not parts:
        return None, None
    ref = np.concatenate(parts)[: int(REF_MAX_S * TARGET_SR)]
    return ref, " ".join(words)


# ---------------------------------------------------------------------------
# BACKENDS -- each returns 16 kHz float32, or raises
# ---------------------------------------------------------------------------


class Vocoder:
    name = "vocoder"
    needs_text = False
    install = "(already available via transformers)"

    def __init__(self, device):
        from transformers import SpeechT5HifiGan, SpeechT5FeatureExtractor
        self.voc = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan").to(device).eval()
        # The mel config must match what the vocoder was trained on.
        self.fe = SpeechT5FeatureExtractor.from_pretrained(
            "microsoft/speecht5_tts")
        self.device = device

    def __call__(self, wav, text, ref, ref_text):
        import torch
        with torch.no_grad():
            mel = self.fe(audio_target=wav, sampling_rate=TARGET_SR,
                          return_tensors="pt")["input_values"].to(self.device)
            return self.voc(mel.squeeze(0)).cpu().numpy().astype(np.float32)


class XTTS:
    name = "xtts"
    needs_text = True
    install = ("pip install coqui-tts 'transformers<5' -- coqui-tts needs transformers 4.x, but OmniVoice pulls 5.x, so they need SEPARATE venvs")

    def __init__(self, device):
        from TTS.api import TTS as CoquiTTS
        self.tts = CoquiTTS(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ).to(device)
        self.sr = 24000

    def __call__(self, wav, text, ref, ref_text):
        import tempfile, os
        fd, refpath = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(refpath, ref, TARGET_SR, subtype="PCM_16")
            # language="es": XTTS has no Cebuano, but its BPE tokenizer
            # is shared across languages and Cebuano orthography is
            # Spanish-derived. This is the bet.
            out = self.tts.tts(text=text, speaker_wav=refpath,
                               language="es")
            return resample(np.asarray(out, dtype=np.float32), self.sr)
        finally:
            try:
                os.unlink(refpath)
            except OSError:
                pass


class OmniVoice:
    name = "omnivoice"
    needs_text = True
    install = "pip install omnivoice"

    def __init__(self, device, model_id="k2-fsa/OmniVoice", lang="ceb"):
        from omnivoice import OmniVoice as OV
        # OmniVoice is a HF PreTrainedModel: OmniVoice(config) needs a
        # config, so construction goes through from_pretrained.
        self.m = OV.from_pretrained(model_id).to(device).eval()
        self.sr = 24000
        self.lang = lang
        try:
            ids = self.m.supported_language_ids()
            if lang not in ids:
                print(f"    WARNING: language {lang!r} not in supported "
                      f"ids ({len(ids)} total) -- output may be unusable")
            else:
                print(f"    language {lang!r} supported")
        except Exception:
            pass

    def __call__(self, wav, text, ref, ref_text):
        import tempfile, os
        fd, refpath = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(refpath, ref, TARGET_SR, subtype="PCM_16")
            # generate() returns a LIST of arrays, one per text.
            out = self.m.generate(
                text=text,
                language=self.lang,
                ref_audio=refpath,
                ref_text=ref_text,
            )[0]
            out = np.asarray(out, dtype=np.float32).squeeze()
            return resample(out, self.sr)
        finally:
            try:
                os.unlink(refpath)
            except OSError:
                pass


BACKENDS = {b.name: b for b in (Vocoder, XTTS, OmniVoice)}


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Audition synthesis backends on the same clips.")
    ap.add_argument("--bonafide-root", default="data/processed/bonafide")
    ap.add_argument("--transcripts", default="transcripts")
    ap.add_argument("--out", default="data/audition")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--speakers", type=int, default=5,
                    help="spread the clips over this many speakers")
    ap.add_argument("--backends", default="vocoder,xtts,omnivoice")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps the GPU free for training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.bonafide_root)
    out_root = Path(args.out)

    if not root.is_dir():
        print(f"ERROR: not found: {root}", file=sys.stderr)
        return 1

    texts = load_transcripts(Path(args.transcripts))
    print(f"transcripts loaded: {len(texts)}")

    rng = random.Random(args.seed)
    speakers = sorted(d for d in root.iterdir() if d.is_dir())
    rng.shuffle(speakers)
    speakers = speakers[: args.speakers]

    picks = []
    per = max(1, args.clips // max(len(speakers), 1))
    for spk in speakers:
        clips = sorted(spk.glob("*.wav"))
        if len(clips) < 2:
            continue
        for p in rng.sample(clips, min(per, len(clips))):
            picks.append((spk, p, clips))
        if len(picks) >= args.clips:
            break
    picks = picks[: args.clips]

    print(f"auditioning {len(picks)} clips "
          f"across {len({s for s,_,_ in picks})} speakers\n")

    # --- load backends, skipping the ones that are not installed ---
    live = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        cls = BACKENDS.get(name)
        if cls is None:
            print(f"  unknown backend {name!r}, skipping", file=sys.stderr)
            continue
        try:
            print(f"  loading {name} ...")
            live.append(cls(args.device))
            print(f"    ok")
        except Exception as e:
            print(f"    SKIPPED {name}: {type(e).__name__}: {e}")
            print(f"    install with: {cls.install}")
    print()

    if not live:
        print("No backends available.", file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    notes = ["AUDITION -- what each clip should say",
             "=" * 60, ""]
    fails = {}

    for spk, src, clips in picks:
        wav, sr = sf.read(src, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = resample(wav, sr)

        text = texts.get(src.stem)
        ref, ref_text = build_reference(clips, exclude=src, texts=texts)

        d = out_root / spk.name
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{src.stem}.ORIGINAL.wav", wav, TARGET_SR,
                 subtype="PCM_16")

        notes.append(f"{spk.name}/{src.stem}")
        notes.append(f'    "{text}"' if text else "    (no transcript)")

        for b in live:
            if b.needs_text and not text:
                notes.append(f"    {b.name}: SKIPPED (no transcript)")
                continue
            if b.needs_text and ref is None:
                notes.append(f"    {b.name}: SKIPPED (no reference audio)")
                continue
            try:
                out = b(wav, text, ref, ref_text)
                sf.write(d / f"{src.stem}.{b.name}.wav", out, TARGET_SR,
                         subtype="PCM_16")
                ratio = len(out) / max(len(wav), 1)
                notes.append(f"    {b.name}: ok  "
                             f"len {len(out)/TARGET_SR:.2f}s "
                             f"(orig {len(wav)/TARGET_SR:.2f}s, "
                             f"ratio {ratio:.2f})")
            except Exception as e:
                fails.setdefault(b.name, []).append(f"{src.stem}: {e}")
                notes.append(f"    {b.name}: FAILED {type(e).__name__}: {e}")
        notes.append("")
        print(f"  {spk.name}/{src.stem}")

    (out_root / "AUDITION.txt").write_text("\n".join(notes), encoding="utf-8")

    print()
    print("=" * 66)
    print(f"Wrote {out_root.resolve()}")
    print(f"Read  {(out_root / 'AUDITION.txt').resolve()}")
    print("=" * 66)
    print()
    print("Listen to each <stem>.ORIGINAL.wav against its siblings:")
    print()
    print("  words    must be intelligible and correct Cebuano. This is")
    print("           what disqualifies a text-driven backend.")
    print("  voice    should resemble the original speaker.")
    print("  texture  slightly synthetic is GOOD -- that is the artefact.")
    print()
    print("Note: a length ratio far from 1.00 is expected for the")
    print("text-driven backends (they re-speak the sentence) but NOT for")
    print("the vocoder, which must stay near 1.00 -- anything else means")
    print("its mel config is wrong.")

    if fails:
        print()
        for name, errs in fails.items():
            print(f"  {name}: {len(errs)} failure(s); first: {errs[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
