# What to add to the methodology

Plain-language companion to `METHODOLOGY_AUDIT.md`. That file is the
technical evidence; this one is what to actually write, and what to do.

Draft sentences below are starting points — rewrite them in your voice.

---

## Part 1 — What your proposal doesn't say yet

### A. The preprocessing chain (new section in 4.3.2)

**What happened.** Your corpus was assembled from three sources that
each arrived in a different state. ElevenLabs returned MP3 at 44.1 kHz;
Meta MMS and the bonafide corpus were 16 kHz WAV. Levels differed too —
TTS engines normalise their output, field recordings don't.

**Why it matters.** Those differences line up almost perfectly with the
class label, so a model can learn "which pipeline made this file"
instead of "is this speech synthetic."

**What you did.** Every clip now goes through one identical chain:
decode → 16 kHz mono → trim silence → MP3 encode/decode → trim again →
loudness-normalise to −26 LUFS → dither → write 16-bit PCM WAV.

Two points worth explaining rather than just listing:

- The **MP3 pass** is not augmentation. Decoding an MP3 to WAV changes
  the container but leaves the compression artefacts in the samples, so
  the only way to stop "has been through MP3" from being a label is to
  put everything through it once.
- **Order matters.** Loudness is normalised last because the codec
  changes level. Silence is trimmed on both sides of the codec because
  the encoder adds its own padding.

> *Draft:* All audio was decoded to 16 kHz mono, silence-trimmed at a
> −40 dB relative threshold, passed through an identical MP3
> encode/decode cycle to equalise codec history across sources,
> loudness-normalised to −26 LUFS (ITU-R BS.1770-4), dithered, and
> written as 16-bit PCM.
> The chain was applied identically to bonafide and spoof recordings.

**Get the unit right.** It is **LUFS**, not dBFS. The pipeline uses
gated loudness per ITU-R BS.1770-4 (via pyloudnorm), which excludes
quiet frames from the measurement, not a raw amplitude average.
Verified by measuring the corpus: whole-file RMS still varies
(bonafide sd 1.26 dB, range −29.8 to −26.1), which would be impossible
under RMS normalisation and is exactly what gating produces.

That gating is also why `rms_db` still separates the classes at EER
0.1963 after normalisation. BS.1770 equalises *speech* loudness;
bonafide carries more pause content, so its whole-file RMS sits about
1.5 dB below the two TTS sources. Worth one sentence, because a
reviewer looking at the feature table will ask why loudness
normalisation left a loudness cue.

### B. Text handling for the two TTS systems (new)

**What happened.** MMS received normalised text — diacritics stripped,
punctuation removed, lowercased — because the model cannot pronounce
diacritics. ElevenLabs received the original transcript.

**Why it matters.** Diacritics in this corpus are stress marks, so
removing them doesn't change the words. But punctuation drives pauses
and intonation in ElevenLabs. The two systems therefore produce
different *prosody* on the same sentences.

**What to do.** Disclose it, and stop comparing the two systems to each
other. Report them separately.

> *Draft:* Meta MMS required text normalisation — diacritic removal,
> punctuation stripping and lowercasing — as the model cannot render
> diacritics. ElevenLabs received unmodified transcripts. Because this
> introduces a prosodic difference between the two synthesis
> conditions, results are reported per system and no direct comparison
> between them is drawn.

Also worth one sentence: roughly 438 clips (0.78%) contain numerals in
contexts the pronunciation manifest doesn't cover — times, dates and
currency in news passages. MMS likely omits these.

### C. Corpus validation (entirely new — and this is the strongest part)

This is the section that doesn't exist yet and should.

**The idea in one line:** before trusting the detector's EER, check how
much of the task is solvable *without listening to the speech.*

**How.** Fit a deliberately weak classifier — logistic regression on
thirteen summary statistics like loudness, silence, noise floor and
spectral shape. It has no access to words, phonemes, or voice identity,
so it is structurally incapable of detecting a deepfake. Whatever EER it
reaches is a floor on how much of the task is channel rather than
speech.

**What you found.** Before normalisation, the channel-only baseline hit
**2.50% EER** — better than the detector's 4.35%. After normalisation it
rose to roughly **9%**, and the remaining separability is dominated by
crest factor, which is a genuine property of synthesised speech rather
than a file artefact.

**Why it's worth a whole section.** Most published work in this area
does none of this. The canonical paper on the problem — Müller et al.,
*"Speech is Silver, Silence is Golden"* (ASVspoof 2021 Workshop) — found
that ASVspoof 2019 models were partly reading silence duration, and that
trimming silence properly moved EER from 3.6% to 15.5%. You measured
your own version of that problem and fixed it. That is a contribution,
not an admission.

> *Draft:* To verify that detector performance reflects synthesis
> artefacts rather than corpus production differences, a channel-only
> baseline was constructed: logistic regression over thirteen
> handcrafted signal statistics containing no linguistic information.
> This baseline achieved X% EER before preprocessing normalisation and
> Y% after, establishing a floor against which detector performance is
> interpreted.

### D. Evaluation protocol (additions to 4.10)

Three changes.

**1. Leave-one-system-out.** Your split is speaker-disjoint on the
bonafide side, but the MMS voice is a single synthetic speaker that
appears in train, validation and test. A detector can memorise that one
timbre. The fix is to hold each system out of training entirely and
measure the drop:

- train on ElevenLabs only → test on MMS
- train on MMS only → test on ElevenLabs

The result you report is the **gap** between matched and unseen test
sets on the same checkpoint. Small gap means the model learned
synthesis. Large gap means it memorised voices. This mirrors ASVspoof,
whose evaluation attacks are deliberately disjoint from its training
attacks.

**2. Stop reporting accuracy.** Your ACC has been pinned at 0.4894 every
single run, which is exactly your bonafide proportion — the model is
assigning every sample to one class at the default threshold. That's
normal for OC-Softmax, whose scores aren't calibrated around zero.
Report EER (threshold-free), and if you want accuracy, report it at the
EER threshold and say so.

**3. Per-system EER.** Report MMS and ElevenLabs separately. The
ElevenLabs number is the more meaningful one, because those clones are
genuinely speaker-disjoint across splits.

### E. Limitations (expand 1.4)

Six items, most of them new:

1. **Meta MMS is a single voice.** One synthetic speaker across all
   28,537 MMS clips, present in train, validation and test. Speaker-
   disjointness protects the bonafide side only.
2. **MMS has no prosodic variation** — fixed speaking rate of 0.95 for
   every utterance, so it is uniformly slower than the human original.
3. **Clone references vary per speaker.** Reference selection stops once
   total duration passes 12 seconds, so some voices were cloned from one
   reference clip and others from five. Clone fidelity therefore varies
   by an uncontrolled amount.
4. **The two synthesis systems are not comparable** — disjoint speakers,
   disjoint prompts, different text normalisation.
5. **Two systems only.** "Generalises to an unseen system" here means
   one specific unseen system, not unseen systems in general.
6. **~438 clips contain unhandled numerals** (0.78% of the corpus).

### F. Reproducibility note

MMS generation seeds once at module import and skips existing files on
resume, so a run interrupted and restarted produces different audio than
an uninterrupted one. Either drop any "fully reproducible" claim for
MMS, or regenerate in a single pass and state that.

### G. Numbers to fix in the existing document

- **10 s → ~4 s.** Section 4.3.2 says clips are padded to 10 seconds
  "per the DeepFense default." The actual value used is 64,600 samples
  (~4.04 s), the ASVspoof convention. The real DeepFense default is
  64,000.
- **Table 3 counts.** The table says 62,433 clips / 152 speakers /
  63h32m. The built corpus is 56,286 bonafide clips across 140 speakers.
  Reconcile, and say what was excluded and why.
- **Spoof clip arithmetic.** MMS (28,537) + ElevenLabs (28,116) = 56,653,
  which exceeds the 56,286 bonafide clips their speakers should account
  for. 367 clips unexplained — see step 2 below.

---

## Part 2 — What to do, in order

### Right now (while training runs)

**Step 1 — Check the ElevenLabs sample rates.** *(10 minutes)*

The audit suspects `trim_silence.py` was only ever run on bonafide and
meta-mms, never ElevenLabs. Check what's actually on disk:

```powershell
Get-ChildItem data\processed\elevenlabs -Recurse -Filter *.wav | Get-Random -Count 20 | ForEach-Object { ffprobe -v error -show_entries stream=codec_name,sample_rate,bits_per_raw_sample -of csv=p=0 $_.FullName }
```

Do the same for `bonafide` and `meta-mms`. If the three don't match,
that's confirmation — and it's already fixed by the normalisation, so
this is for the write-up, not for panic.

**Step 2 — Reconcile the 367 extra clips.** *(20 minutes)*

Check whether the two speaker lists overlap:

```powershell
Import-Csv manifests\mms_selected_speakers.csv | Select-Object -Expand speaker_id | Sort-Object > mms.txt
Import-Csv manifests\elevenlabs_selected_speakers.csv | Select-Object -Expand speaker_id | Sort-Object > el.txt
Compare-Object (Get-Content mms.txt) (Get-Content el.txt) -IncludeEqual | Where-Object SideIndicator -eq '=='
```

Any output means a speaker was assigned to both systems. If the lists
are clean, the excess is stale files from earlier test runs — check for
`.2.wav` files whose stem appears in `elevenlabs-reference\`.

This one matters because it means some spoof clips have no bonafide
counterpart, which breaks the text pairing.

**Step 3 — Listen to three MMS clips with numbers in them.** *(5 minutes)*

Find an utterance whose transcript contains a bare digit, and play the
MMS version. Confirm whether the number is spoken or silently dropped.
Three clips is enough to know.

### When the 2-epoch run finishes

**Step 4 — Read epoch 2 and 3.** Compare to the ~9% channel-only floor.
Well below it means the detector is doing real work. Around it means
stop and diagnose before committing to more runs.

**Step 5 — Decide the experiment matrix.** Four runs at roughly 2.5
hours per epoch. My suggestion:

| run | trains on | answers |
|---|---|---|
| main | both codecs, both systems | headline number |
| noaug | no augmentation | what codec augmentation buys |
| loso-el | ElevenLabs only | generalises to an unseen system? |
| loso-mms | MMS only | generalises to an unseen system? |

Cross-codec testing doesn't need its own training runs — score one
checkpoint against clean, Opus and AMR test sets.

### Writing

**Step 6 — Write section C first.** The corpus validation section is the
most defensible thing you have and it's freshest in your head. It also
sets up everything else: once the reader knows you measured the channel
floor, your EER means something specific.

**Step 7 — Then the limitations.** Six items from section E. Write them
plainly. A measured limitation reads as rigour; an unexamined one reads
as an oversight a reviewer found first.

**Step 8 — Then fix the numbers in G.** Mechanical, do it last.

---

## The one-paragraph version

Your proposal describes building a Cebuano deepfake detector. What
actually happened is that you built one, discovered its performance came
mostly from production artefacts rather than speech, measured exactly
how much, fixed the preprocessing, and re-measured. That's a better
thesis than the one you proposed. The methodology needs to say so.
