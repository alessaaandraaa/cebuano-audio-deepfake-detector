# Methodology audit — preprocessing and generation

Findings from reading `preprocessing/` against the thesis methodology,
after the channel-leak investigation of 2026-09-16.

**Read this first.** The normalisation pass (`training/normalise_corpus.py`)
fixes *channel* asymmetries — loudness, container, sample rate, silence
duration, noise floor, bit depth. Everything below is either a *content*
or *design* asymmetry, which no amount of audio post-processing can
undo. They are separate problems and the normalisation does not touch
them.

Each finding is marked **VERIFIED** (read in the code) or **INFERRED**
(reasoned from the code but not directly confirmed). Verify the inferred
ones before citing them.

---

## Tier 1 — content confounds

These are the ones that undermine the claim that the detector
distinguishes synthesis from speech, because the spoof and bonafide
utterances do not say the same thing.

### 1.1 MMS and ElevenLabs receive different text — VERIFIED, but the effect is PROSODIC, not lexical

**Corrected 2026-09-16 after measuring.** The first version of this
finding said MMS was "silently given different words". That was wrong
and is retracted. Measured over 20,961 transcript lines (50 of 140
speakers):

`generate_elevenlabs.py:108` reads `transcripts/` (raw).
`generate_mms.py:15-19` reads `transcripts_mms/` (normalised).

| | MMS input | ElevenLabs input | prevalence |
|---|---|---|---|
| diacritics | stripped | intact | **50.88% of lines** |
| punctuation | replaced with spaces | intact | most lines |
| case | lowercased | original | most lines |

**Diacritics are stress marks, not lexical content.** `págtagád` and
`pagtagad` are the same word; the diacritic is a reading aid in this
corpus. Stripping them removes a pronunciation hint. MMS cannot
pronounce them at all, so stripping was a correct engineering decision,
not an oversight.

**What remains true, and is narrower:** ElevenLabs received the stress
marks and the punctuation; MMS received neither. Punctuation is the
primary pause and intonation control for `eleven_v3`. So the two spoof
sources differ systematically in *prosody* while saying the same words.

**Consequence:** an MMS-vs-ElevenLabs EER gap cannot be attributed to
the engines. It is already confounded by disjoint speaker sets and
disjoint prompt sets (see 4.3); text normalisation is a third strand.
Report the two conditions separately; do not compare them head to head.

**Action: disclosure, not code.** State in 4.3.2 that MMS required
diacritic stripping and punctuation removal for intelligibility, that
ElevenLabs received unmodified text, and that the two conditions are
therefore reported separately.

### 1.2 Digits outside Random Digit entries — VERIFIED, and it is 0.78% of the corpus

**Corrected 2026-09-16 after measuring.** The first version of this
finding implied a broad content confound. Measured, it is a footnote.

The three-tier number handling in `extract_transcripts.py` is sound and
works:

- **1–9 in Random Digit entries** (255-291) — per-utterance lookup in
  `original_pronunciations.tsv`, keyed on wav filename, with a mismatch
  warning. This is the *tulo* vs *tres* problem, correctly solved.
- **10–100, all sources** (302-306) — fixed corpus-wide mapping.
  **Verified effective: zero standalone 10–100 numbers remain in the
  transcripts.**
- **1–9 outside Random Digit entries** (293-300) — counted, not
  expanded. This is the gap, and the code comment says so deliberately.

Measured prevalence over 20,961 lines:

| | lines | % |
|---|---|---|
| bare 1–9 | 45 | 0.21% |
| digits inside tokens (`PO2`, `9mm`, `DT 125`) | 81 | 0.39% |
| currency (`P350.00`, `P12,000`) | 29 | 0.14% |
| times (`alas 9:00`) | 18 | 0.09% |
| **any digit in text** | **163** | **0.78%** |

Extrapolated: **~438 clips of 56,286.**

All of them are news-passage reads (*"pasado ala 1 kagahapon"*,
*"Pebrero 8 sa hapon"*, *"Kilometro 5"*), never Random Digit entries —
which is precisely why the pronunciation manifest does not cover them.

For those ~438 clips, `normalize_mms.py:515` preserves the digit
(`\w` matches digits) and `generate_mms.py:346` passes `normalize=True`
to a character-level VITS with no digit glyphs, which drops unknown
characters silently. So MMS likely says nothing where the human said a
number. **INFERRED** — confirm by listening to two or three.

**Action, optional:** MMS regeneration is free, so either exclude these
~438 utterances or extend the mapping to cover times and currency.
Either is fine. At 0.78% it is a limitation sentence, not a blocker.

## Tier 2 — asymmetric processing

### 2.1 `trim_silence.py` uses an absolute threshold, and may never have run on ElevenLabs — VERIFIED code, INFERRED invocation

`trim_silence.py:88-92,141` gates on `silence_thresh=-40.0` **absolute
dBFS**, not relative to each file's own level. Studio read speech, VITS
output and service-normalised ElevenLabs sit at different levels, so
identical parameters trim different amounts per class.

Worse, it looks like it was not applied uniformly:

- `trim_silence.py:30-36` usage block shows only `bonafide` and `meta-mms`
- `tests/check_leading_silence.py:8-9` — same two
- every downstream consumer points at `data/processed/{bonafide,meta-mms,elevenlabs}`,
  **not** at any `-trimmed` directory

**This is the single highest-value thing to verify.** Either ElevenLabs
was never trimmed, or the trimmed trees are not what got manifested.

Two more issues in the same file:

- **Silent fallback**, lines 94-96: `if not nonsilent_ranges: return audio`.
  An all-silence-by-threshold file passes through untrimmed and is
  counted as "unchanged", indistinguishable from a file that had no
  silence. Quiet MMS output is the likeliest population to hit this, so
  untrimmed files cluster in one class.
- **Bit-depth change**: `export(format="wav")` via pydub yields 16-bit
  PCM, while `resample_dataset.py`'s `sf.write` on a librosa float32
  array yields 32-bit float. Trimming changes the container subtype, so
  trimmed and untrimmed sources differ in bit depth.

*Mitigation already in place:* `normalise_corpus.py` re-trims everything
with a **relative** threshold (40 dB below each file's own peak frame),
which fixes this going forward. What it cannot undo is real audio that
the earlier absolute-threshold pass removed from some classes and not
others.

### 2.2 ElevenLabs never specified `output_format` — VERIFIED

`generate_elevenlabs.py:563-573` sets `Accept: audio/wav` in the header
and nothing else. The ElevenLabs endpoint selects container and rate via
an `output_format` **query parameter**, not the `Accept` header. Grep
for `output_format|sample_rate|44100|16000|resample` across
`preprocessing/elevenlabs/` returns nothing. Bytes are written verbatim
under a `.2.wav` name (`788-794`).

This is the origin of the MP3-vs-WAV leak: the extension was asserted by
the script, not by the payload.

*Mitigation already in place:* normalisation re-decodes and rewrites
everything uniformly. For the write-up, the root cause belongs in the
methodology, not just the fix.

### 2.3 `resample_dataset.py`: a missing `.log` silently disables exclusion — VERIFIED

Lines 145-152: no log file → `continue`, and that speaker's
`TGL_`/`CEB_Utt_Eng` clips are all resampled into bonafide. Meanwhile
`extract_transcripts.py:431-434` skips the speaker entirely, so there is
no transcript and therefore no spoof. One missing log injects a few
hundred bonafide-only, partly-Tagalog clips with no spoof counterpart.
The warning goes to stderr among thousands of lines.

Minor, same file: `build_exclusion_map` iterates only top-level dirs
(141) while `find_audio_files` uses `rglob` (171), so audio nested deeper
can never be excluded. And `EXCLUDE_SOURCE_PREFIXES` is defined in three
places (`resample_dataset.py:47`, `extract_transcripts.py:68`,
`inventory_bonafide.py:53`) — currently identical, but any future edit to
one desynchronises the audio set from the transcript set.

---

## Tier 3 — pairing and provenance

### 3.1 367 more spoof clips than the spoof speakers have bonafide clips — VERIFIED arithmetic

From `manifests/splits/split_summary.txt`:

| | train | val | test | total |
|---|---|---|---|---|
| bonafide | 39351 | 8303 | 8632 | **56286** |
| meta-mms | 20295 | 4108 | 4134 | **28537** |
| elevenlabs | 19064 | 4553 | 4499 | **28116** |

Bonafide reconciles perfectly: 56,526 included clips − 240 moved to
`elevenlabs-reference/` = 56,286.

But if the speaker manifests partition the 140 speakers, MMS and
ElevenLabs cover disjoint halves, so `mms + el ≤ bonafide`. Instead
28,537 + 28,116 = **56,653** — 367 too many. Since MMS is bounded by its
speakers' clips, the excess is on the ElevenLabs side, which also skipped
240 references, so it over-generated by roughly 600 relative to what
should exist.

Candidates, all checkable:

1. Speaker overlap between `mms_selected_speakers.csv` and
   `elevenlabs_selected_speakers.csv`. 367 is about one heavy speaker's
   worth (max 450 clips per `audio_summary.txt`).
2. `regenerate_elevenlabs.py` writing reference-clip spoofs (see 3.2).
3. Stale outputs. Neither generator deletes; both treat "file exists" as
   success. Files from earlier `--limit` runs persist and get swept into
   the parquets.
4. Utterances in the `.log` whose wav is missing on disk — generation is
   transcript-driven, bonafide is filesystem-driven.

Whatever the cause, some spoof clips have no bonafide twin, which breaks
the text pairing that is the main defence against a content confound.

### 3.2 `regenerate_elevenlabs.py` has none of the main script's guards — VERIFIED

`generate_elevenlabs.py:720-728` skips any utterance whose filename is a
clone reference. `regenerate_elevenlabs.py` has no such check — its only
gate is `if speaker_id not in voice_ids` (464-474). It does not consult
`elevenlabs_selected_speakers.csv` or `elevenlabs-reference/`, and writes
straight to `OUTPUT_ROOT/<speaker>/<stem>.2.wav` (480-526).

If any row of the regeneration TSV names a wav that
`select_reference_clips.py` moved into `elevenlabs-reference/`, you get a
spoof file with no bonafide counterpart, whose source clip is one of the
clips the voice was cloned from.

It also uses a **third** text source — the `saying` column of the TSV
(line 500) — neither `transcripts/` nor `transcripts_mms/`.

### 3.3 MMS has no `--overwrite`, so its audio goes stale — VERIFIED

`generate_mms.py:588-592` skips existing files and there is no
`--overwrite` flag in the parser at all. `generate_elevenlabs.py:902-909`
has one, and `regenerate_elevenlabs.py` exists solely to re-derive
ElevenLabs utterances.

So any correction to `transcripts_mms/` after the first run is silently
not reflected in the MMS audio. Given that a whole script exists to
retro-fix ElevenLabs digit utterances (its example rows are exactly
`1 → uno`, `2 → dos`), the presumption is that **ElevenLabs was corrected
and MMS was not** — meaning MMS audio does not match its own labelled
transcript for those utterances.

**Check:** compare mtimes of `transcripts_mms/` against
`data/processed/meta-mms/`. Any transcript newer than its audio is stale.

### 3.4 Clone reference selection varies wildly per speaker — VERIFIED

`select_reference_clips.py` docstring promises "3-5 utterances at least
`--min-duration` seconds long", default 10.0. The parser has **no
`--min-duration`**; it uses `--min-total-duration` default 12.0 (150-158)
and `--min-per-speaker` default **1** (161-167).

`select_toward_total` (71-129) stops as soon as one clip clears 12 s. So
one speaker is cloned from a single reference and another from five.
Clone fidelity — and therefore how hard that speaker's spoof is to
detect — varies by an uncontrolled amount.

**Worth doing:** record per-speaker reference duration alongside
per-speaker EER. If they correlate, that is a confound in the ElevenLabs
results, and it is a nice analysis either way.

### 3.5 `clone_voice.py` denoising flag — VERIFIED, conditional

`clone_voice.py:502,530-533,1223-1224` expose `remove_background_noise`,
default `False`. If it was ever passed, the clones were built from
denoised references, so every ElevenLabs clip inherits a cleaner noise
floor than the bonafide recordings of the same speaker — exactly the
channel cue you are eliminating. Check shell history; disclose or rebuild.

---

## Tier 4 — reproducibility and disclosure

### 4.1 MMS generation is not reproducible across resumed runs — VERIFIED

`generate_mms.py:105-107` calls `set_seed(42)` once at module import. The
resume check (588) `continue`s **before** generation, so the RNG stream
position at utterance *k* depends on how many utterances that particular
process actually generated. A run resumed three times produces different
audio than an uninterrupted run, and the first utterance of each session
shares an identical RNG start state.

Not a class-separability threat, but it invalidates any "seed 42, fully
reproducible" claim.

### 4.2 MMS has zero speaking-rate variance — VERIFIED

`generate_mms.py:358-362` fixes `speaking_rate = 0.95`,
`noise_scale = 0.333`, `noise_scale_duration = 0.4` for every utterance.
So MMS is systematically slower than the human original and has no tempo
variation at all — a duration cue. Partly masked by the 64,600-sample
random crop, but not for clips shorter than ~4 s, which get
repetition-padded instead.

### 4.3 MMS single voice spans all three splits — VERIFIED

Already known and being addressed by `make_loso_splits.py`. Recording it
here for completeness: `generate_mms.py:99` loads a single-speaker VITS
with no speaker embedding. All 28,537 MMS clips are the same synthetic
voice, present in train, val and test. The `LEAKAGE CHECK:
speaker-disjoint` line is computed over bonafide speaker IDs and is blind
to it.

Related: you **cannot** attribute an MMS-vs-ElevenLabs EER gap to the
engines, because the two conditions use disjoint speaker sets, disjoint
prompt sets, and (1.1) different text. Triply confounded.

---

## What is symmetric and fine

Briefly, so these do not get re-litigated:

- `trim_silence.py` **as a script** is source-agnostic — identical code
  path for any tree, preserves relative paths, never modifies originals.
  The risk is in how it was invoked, not the logic.
- **Reference-clip leakage into bonafide is correctly prevented.**
  `select_reference_clips.py:261` uses `shutil.move`, not copy, and
  `generate_elevenlabs.py:720` additionally skips generating spoof for
  them. The arithmetic confirms it. The only hole is 3.2.
- **TGL_/CEB_Utt_Eng exclusion is consistent** across the three scripts
  that implement it, so audio and transcripts are excluded together.
  (`audio_summary.txt`'s "Excluded TGL clips" label under-describes the
  counter — it includes CEB_Utt_Eng. Cosmetic.)
- **Number expansion 10–100 and per-utterance 1–9 for Random Digit
  entries** (`extract_transcripts.py:243-285`) is careful work:
  per-recording pronunciation lookup with a mismatch warning is the right
  design. The only gap is the non-Random-Digit case (1.2).
- **`select_speakers.py`** stratification is deterministic and correct.
  `round(len*0.5)` uses banker's rounding, so 81 female → 40 and
  59 male → 30, giving 70/70. Harmless, just non-obvious.
- **ElevenLabs failure tracking** exists, with
  `tests/check_missing_elevenlabs.py` to enumerate gaps. MMS has no
  equivalent completeness check.

---

## Ordered checklist

Cheap verification first — several of these may turn out to be nothing.

1. `comm` the two speaker manifests for overlap; reconcile the 367-clip
   excess. (3.1)
2. `ffprobe` 100 random files from each tree, ignoring extensions, and
   tabulate real container + rate. (2.1, 2.2)
3. Confirm which trees the parquets actually point at — trimmed or
   untrimmed. (2.1)
4. ~~Read `single_digit_skipped_non_random`~~ — done, measured
   directly: 0.78% of lines carry any digit, ~438 clips. Decide
   whether to exclude them or extend the mapping. (1.2)
5. Diff `transcripts_mms/<spk>.txt` against `transcripts/<spk>.txt` for
   20 utterances and listen to the bonafide audio against both. (1.1, 1.3)
6. Check whether any `elevenlabs-reference/` filename has a matching
   `.2.wav` in `data/processed/elevenlabs/`. (3.2)
7. Compare mtimes: `transcripts_mms/` vs `data/processed/meta-mms/`. (3.3)
8. Check shell history for `--remove-background-noise`. (3.5)

## Thesis edits these imply

- **4.3.2** — document the real preprocessing chain: MMS required
  diacritic stripping and punctuation removal for intelligibility while
  ElevenLabs received unmodified text (1.1), and MMS drops
  out-of-vocabulary characters, affecting ~438 clips (1.2, 1.4).
- **Results** — report MMS and ElevenLabs separately and do NOT compare
  them head to head: disjoint speakers, disjoint prompts, different text
  normalisation (1.1, 4.3).
- **1.4 / limitations** — single MMS voice present in all splits (4.3);
  fixed speaking rate (4.2); clone reference duration varying 1–5 clips
  per speaker (3.4); MMS/ElevenLabs comparison is confounded (4.3).
- **Reproducibility statement** — drop or qualify any "seed 42, fully
  reproducible" claim for MMS (4.1).
- **Results** — report per-source EER, and treat the ElevenLabs number as
  the more meaningful one, since those clones are genuinely
  speaker-disjoint across splits.
