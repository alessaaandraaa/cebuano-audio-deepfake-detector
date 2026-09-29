r"""
vocode_copysynth.py

Vocoder copy-synthesis: resynthesize bona fide clips through a neural
vocoder to produce spoofed audio that keeps the ORIGINAL SPEAKER'S
VOICE.

WHY

Meta MMS is a single synthetic voice. That one voice appears in train,
val and test, so a detector can score near-perfectly by recognising a
timbre rather than by detecting synthesis -- measured here at EER
0.000237 against ElevenLabs' 0.012254, a 52x gap, and 0.0000 when a
model is trained on MMS alone.

Copy-synthesis removes the confound structurally. The vocoder takes a
mel spectrogram extracted from a real clip and regenerates the
waveform, so the output carries the same speaker, the same words and
(almost) the same duration and tempo. Voice identity therefore carries
NO information about the label -- there is no timbre to memorise.

    Wang & Yamagishi, "Spoofed training data for speech spoofing
    countermeasure can be efficiently created using neural vocoders",
    ICASSP 2023 (arXiv:2210.10570)

Their argument for why this still represents TTS artefacts: "Since most
TTS and VC systems end with a vocoder or vocoder-like module, the
vocoded data is expected to capture the vocoders' artifacts that affect
TTS and VC systems."

WHY THIS VOCODER

microsoft/speecht5_hifigan is a HiFiGAN operating natively at 16 kHz,
shipped inside transformers, which is already installed. Native rate
matters: Wang & Yamagishi's Voc.v1 needed 16k->24k->16k resampling and
scored 36.57% pooled EER against 16.35% for the 16 kHz configuration.
Resampling round trips cost more than the vocoder choice does.

A CAVEAT YOU MUST WRITE DOWN

This vocoder is pretrained on English (LibriTTS), NOT fine-tuned on
Cebuano. That is Wang & Yamagishi's Voc.v2 configuration, which was
their WORST at 39.95% pooled EER; their best (v4) fine-tuned the
vocoders on the target bona fide data and reached 16.35%.

That gap is about generalising to unseen attacks, which is not what
this is for here -- the goal is removing the speaker confound, and the
copy-synthesis STRUCTURE does that regardless of vocoder fidelity. But
state it in limitations, and cite v4 as the stronger configuration you
lacked the compute for.

Vocoders are language-agnostic: mel spectrogram in, waveform out. No
text front-end, no phoneme set. Cebuano is irrelevant to them, which is
exactly why this works where a TTS would not.

USAGE

    # listen first -- 8 clips, originals written alongside
    python preprocessing\vocode_copysynth.py --limit 8 ^
        --out data\vocoder_test --side-by-side

    # then the real pass
    python preprocessing\vocode_copysynth.py ^
        --bonafide-root data\processed\bonafide ^
        --out data\processed\vocoder-hifigan --suffix .3
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import torch
except ImportError:
    print("Missing dependency. Run: pip install soundfile torch",
          file=sys.stderr)
    sys.exit(1)

TARGET_SR = 16000


def load_vocoder(model_id: str, device: str):
    """HiFiGAN plus the feature extractor that makes its input mels."""
    try:
        from transformers import SpeechT5HifiGan, SpeechT5FeatureExtractor
    except ImportError:
        print("Needs transformers. Run: pip install transformers",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_id} ...")
    voc = SpeechT5HifiGan.from_pretrained(model_id).to(device).eval()

    # The mel config MUST match what the vocoder was trained on. Using
    # the matching feature extractor is the whole reason this works;
    # hand-rolled mels with a different hop or fmax produce audio that
    # sounds plausible and is subtly wrong.
    fe = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")

    return voc, fe


@torch.no_grad()
def copy_synth(wav, fe, voc, device):
    """bona fide waveform -> mel -> vocoder -> spoof waveform."""
    mel = fe(
        audio_target=wav,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )["input_values"].to(device)

    out = voc(mel.squeeze(0)).cpu().numpy().astype(np.float32)

    return out


def iter_clips(root: Path, limit=None):
    n = 0
    for spk in sorted(d for d in root.iterdir() if d.is_dir()):
        for p in sorted(spk.glob("*.wav")):
            yield spk.name, p
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser(
        description="Vocoder copy-synthesis for spoof generation."
    )
    ap.add_argument("--bonafide-root", default="data/processed/bonafide")
    ap.add_argument("--out", required=True,
                    help="output root; <out>/<speaker>/<stem><suffix>.wav")
    ap.add_argument("--suffix", default=".3",
                    help="synthesis suffix (.1 mms, .2 elevenlabs)")
    ap.add_argument("--model", default="microsoft/speecht5_hifigan")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N clips -- use this to audition first")
    ap.add_argument("--side-by-side", action="store_true",
                    help="also copy the original next to each output, so "
                         "you can A/B them in one folder")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps the GPU free for training")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.bonafide_root)
    out_root = Path(args.out)

    if not root.is_dir():
        print(f"ERROR: not found: {root}", file=sys.stderr)
        return 1

    voc, fe = load_vocoder(args.model, args.device)

    print(f"  bonafide : {root}")
    print(f"  output   : {out_root}  (suffix {args.suffix})")
    print(f"  device   : {args.device}")
    print()

    n = skipped = 0
    dur_err, peaks, rmss = [], [], []
    t0 = time.time()

    for spk, src in iter_clips(root, args.limit):
        dst_dir = out_root / spk
        dst = dst_dir / f"{src.stem}{args.suffix}.wav"

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        wav, sr = sf.read(src, dtype="float32")

        if sr != TARGET_SR:
            print(f"  SKIP (sr={sr}): {src}", file=sys.stderr)
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        out = copy_synth(wav, fe, voc, args.device)

        dst_dir.mkdir(parents=True, exist_ok=True)
        sf.write(dst, out, TARGET_SR, subtype="PCM_16")

        if args.side_by_side:
            sf.write(dst_dir / f"{src.stem}.ORIGINAL.wav", wav,
                     TARGET_SR, subtype="PCM_16")

        dur_err.append(len(out) / len(wav) if len(wav) else 0.0)
        peaks.append(float(np.abs(out).max()) if out.size else 0.0)
        rmss.append(float(np.sqrt((out ** 2).mean())) if out.size else 0.0)
        n += 1

        if n % 200 == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n} clips  ({rate:.1f}/s)")

    el = time.time() - t0
    print()
    print(f"Wrote {n} clips in {el/60:.1f} min "
          f"({n/max(el,1e-9):.1f}/s), skipped {skipped} existing.")

    if n:
        d = np.array(dur_err)
        print()
        print("  SANITY -- copy-synthesis should preserve duration:")
        print(f"    length ratio out/in : mean {d.mean():.4f}  "
              f"min {d.min():.4f}  max {d.max():.4f}")
        print(f"    output peak         : mean {np.mean(peaks):.4f}  "
              f"max {np.max(peaks):.4f}")
        print(f"    output rms (dB)     : "
              f"{20*np.log10(np.mean(rmss)+1e-12):.2f}")
        print()
        if abs(d.mean() - 1.0) > 0.05:
            print("    WARNING: durations drifted >5%. The mel hop length "
                  "probably does not match the vocoder. Do not proceed.")
        if np.max(peaks) >= 0.999:
            print("    NOTE: output clipping at full scale. Harmless if you "
                  "peak-normalise downstream, which normalise_corpus.py does.")

    if args.limit:
        print("  This was an AUDITION (--limit). Listen to the output "
              "next to the .ORIGINAL.wav files before committing to a "
              "full pass -- the words and the speaker should be intact, "
              "and it should sound slightly synthetic. That slight "
              "synthetic quality IS the artefact you want.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
