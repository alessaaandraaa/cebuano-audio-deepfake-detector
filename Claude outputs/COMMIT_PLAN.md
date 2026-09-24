# Commit plan

13 commits, ordered so each one leaves the repo in a coherent state and
the history reads as the story of what happened.

Run one block at a time. Check `git status` between blocks if you want
to be careful. Nothing here uses `git add -A`, deliberately.

---

## 0. Before anything

Clear the stale lock (from my timed-out command):

```powershell
Remove-Item .git\index.lock -ErrorAction SilentlyContinue
```

Confirm the ignore rules are live **before** staging anything, so the
18 GB of checkpoints can never slip in:

```powershell
git check-ignore -v outputs/ data/ venv/ .env
```

All four should print a matching rule. If `outputs/` does not, stop —
commit 1 has to land first and it is the one that adds that rule.

---

## 1. chore: ignore training outputs and checkpoints

First, so nothing heavy can enter any later commit.

```powershell
git add .gitignore
git commit -m "chore: ignore training outputs and model checkpoints

Each checkpoint is ~3.8 GB because the file carries Adam optimiser
state alongside the 300M-parameter model, and the trainer writes one
on every validation improvement. outputs/ reached 18 GB after two
short test runs; a 30-epoch run can add 40-60 GB.

Also ignores *.pth/*.pt/*.ckpt as a backstop and the probe feature
caches. manifests/ stays tracked on purpose - 26 MB, and the exact
splits are the reproducibility record behind every reported number."
```

---

## 2. build: pin formatting config

```powershell
git add pyproject.toml
git commit -m "build: pin black config at 79 columns with string processing

Fixes the formatting the refactor in the next commit applies, so the
style is reproducible rather than whatever the local black defaults
happen to be."
```

---

## 3. style: normalise script headers and formatting

The refactor pass: consistent module docstrings, corrected stale
filenames in headers, raw strings for docstrings containing Windows
paths, 79-column wrapping.

```powershell
git add preprocessing/elevenlabs/clone_voice.py preprocessing/elevenlabs/delete_voices.py preprocessing/elevenlabs/generate_elevenlabs.py preprocessing/elevenlabs/regenerate_elevenlabs.py preprocessing/elevenlabs/select_reference_clips.py
git add preprocessing/extract_english_entities.py preprocessing/extract_transcripts.py preprocessing/find_nonrandom_numbers.py preprocessing/inventory_bonafide.py preprocessing/resample_dataset.py preprocessing/select_speakers.py preprocessing/trim_silence.py
git add preprocessing/meta_mms/generate_mms.py preprocessing/meta_mms/normalize_mms.py preprocessing/number_mapping/check_random_digits.py
git add tests/

git commit -m "style: consistent headers and formatting across scripts

Behaviour-preserving. Module docstrings given a uniform shape, three
stale filenames in headers corrected, docstrings containing Windows
paths made raw so backslash escapes stop mangling the text, and
everything wrapped to 79 columns.

Verified unchanged: ASTs identical for all 22 files, --help output
byte-identical, one functional smoke test byte-identical."
```

---

## 4. feat: source audit and verification sampling

```powershell
git add preprocessing/audit_sources.py sampling/select_verification_samples.py
git commit -m "feat: add source audit and verification sampling scripts

audit_sources.py cross-checks the bonafide and spoof trees for
orphans and class asymmetry. Audit-only by default; --apply moves
files to _excluded/ rather than deleting, so nothing is lost.

select_verification_samples.py draws the stratified subset used for
the human spoof-quality validation protocol."
```

---

## 5. feat(training): speaker-disjoint manifest builder

```powershell
git add training/build_manifests.py
git commit -m "feat(training): build speaker-disjoint, gender-stratified splits

70/15/15 by speaker, so no speaker appears in more than one split.
Writes a summary with the per-split gender balance and an explicit
leakage check over all three split pairs.

Paths are absolute, so this must run on the training machine."
```

---

## 6. feat(training): DeepFense integration

```powershell
git add training/codec_opus_amr.py training/deepfense_patches.py training/run_training.py
git commit -m "feat(training): DeepFense integration - codecs, patches, runner

codec_opus_amr.py registers opus_codec and amrnb_codec transforms
implementing the bitrate policy in the methodology. The built-in
codec transform exposes only noise_ratio and gives no control over
codec or bitrate. require_encoder() raises rather than silently
no-opping, so a missing ffmpeg encoder fails loudly.

deepfense_patches.py fixes three upstream problems:
  1. build_transforms_pipeline returns unpicklable closures, which
     kills Windows spawn DataLoader workers
  2. build_dataloader passes collate_fn as a lambda, same failure one
     layer deeper
  3. no gradient clipping exists anywhere - loss.backward() runs
     straight into optimizer.step()

Codec augmentation is CPU-bound at ~368 ms/clip, so losing workers is
the difference between a multi-day and a multi-week run.

run_training.py imports the transforms (registration is a decorator
side effect) and applies the patches before handing off to the
DeepFense CLI. The bare CLI fails with KeyError: 'opus_codec'."
```

---

## 7. feat(training): experiment configs

```powershell
git add training/configs/
git commit -m "feat(training): add training configs

smoke / smoke_unfrozen for pipeline and VRAM checks, timing_2epoch
for short runs at the real learning rate, train_main for the full run.

Schema notes are inline, verified against the installed package
rather than the README, which is wrong about dataset_type,
dataset_names and whether transform args are nested.

timing_2epoch sets T_max: 30 while running 5 epochs on purpose:
cosine decay is defined over T_max, so matching it to a short epoch
count collapses the LR to 25% by epoch 3 and the test stops
resembling the real run."
```

---

## 8. feat(training): channel-leak probe

The measurement that changed the project.

```powershell
git add training/probe_shortcuts.py
git commit -m "feat(training): add channel-only shortcut probe

Fits logistic regression on 13 handcrafted signal statistics with no
linguistic content - loudness, silence, noise floor, spectral shape.
Structurally incapable of detecting synthesis, so whatever EER it
reaches is a floor on how much of the task is solvable without
listening to the speech.

On the original corpus it reached 0.0250 EER against the detector's
0.0435: a model that cannot hear words beat the detector. It also
found the cause - 4041/8000 sampled spoof clips were MP3 at 44.1 kHz
and zero bonafide clips were.

Also reports file provenance by class and per-feature separability,
so a leak is named rather than just flagged.

--splits-dir is required with no default: it previously defaulted to
manifests/splits, which silently measured the pre-normalisation
corpus and returned a plausible wrong answer."
```

---

## 9. feat(training): corpus normalisation

The fix.

```powershell
git add training/normalise_corpus.py
git commit -m "feat(training): normalise every clip through one identical chain

16 kHz mono -> trim silence -> MP3 round trip -> trim -> loudness
normalise -> dither -> 16-bit PCM WAV, applied identically to both
classes.

The MP3 pass is equalisation, not augmentation: decoding an MP3 to
WAV changes the container but leaves the compression artefacts in the
samples, so the only way to stop 'has been through MP3' from being a
label is to put everything through it once.

Ordering matters. Loudness is last because the codec changes level,
and silence is trimmed on both sides of the codec because the encoder
adds its own padding. The trim threshold is relative to each file's
own peak, not absolute, so it removes the same proportion regardless
of how loud the source was.

Result: channel-only probe EER on the test split went 0.0665 ->
0.1345. Originals are untouched; output goes to a parallel tree so
the before/after comparison stays reproducible."
```

---

## 10. feat(training): generalisation and ablation tooling

```powershell
git add training/make_loso_splits.py training/make_ablation_configs.py
git commit -m "feat(training): leave-one-system-out splits and ablation configs

The split is speaker-disjoint on the bonafide side, but the Meta MMS
voice is a single synthetic speaker present in train, val and test.
A detector can memorise that timbre and score well on half the spoof
class without learning anything about synthesis, and the leakage
check cannot see it because the repeated thing is not in the speaker
column.

make_loso_splits.py holds each system out of training entirely. The
result is the gap between test_matched and test_unseen on the same
checkpoint. The two test sets share one bonafide pool and carry equal
spoof counts, so the gap is attributable only to the held-out system.

make_ablation_configs.py derives the matrix from one base config,
then diffs each output back against it and refuses to write anything
that differs beyond its declared keys. Hand-copying five YAMLs is how
a stray batch size ends up in a results table with no way to tell
from the numbers."
```

---

## 11. feat(training): threshold-calibrated scoring

```powershell
git add training/score_predictions.py
git commit -m "feat(training): score predictions at a calibrated threshold

DeepFense registers three metrics - EER, ACC, F1_SCORE - and two are
unusable as shipped. ACC and F1_SCORE both hardcode
predictions = (scores > 0), but OC-Softmax scores are not calibrated
around zero, so every sample lands one side of it and ACC collapses
to the class prior. That is the 0.4894 in every run so far. The EER
routine does compute the equal-error threshold, but Evaluator strips
any key containing 'threshold' before returning. There is no
precision or recall metric at all.

This computes accuracy, precision, recall and F1 for both classes
from the saved per-sample scores, at a threshold taken from
validation and applied to test. Omitting --calibrate uses an oracle
threshold and labels it as optimistic.

EER comes with a bootstrap percentile interval; at a few thousand
trials the difference between 3.0% and 3.4% is usually noise.

Verified against synthetic fixtures with analytic EER 0.0668 and
0.1587."
```

---

## 12. data: manifests, splits and mapping tables

```powershell
git add manifests/audio_summary.txt manifests/elevenlabs_selected_speakers.csv manifests/manifest_bonafide.csv manifests/mms_selected_speakers.csv manifests/original_pronunciations.tsv
git add manifests/splits/ manifests/splits_norm/ manifests/splits_smoke/ manifests/loso/
git add preprocessing/ceb_english_mapping.json preprocessing/number_mapping/numbers.json preprocessing/manifests/missing_number_files.txt

git commit -m "data: regenerate manifests and add dataset splits

splits/ is the pre-normalisation corpus and splits_norm/ the
normalised one. Both are kept: the pair is the evidence for the
preprocessing result, and discarding the 'before' makes the
comparison unreproducible.

loso/ holds the leave-one-system-out derivations, splits_smoke/ the
small subset the smoke configs use.

26 MB total, and these files are what tie every reported number to a
specific set of clips."
```

---

## 13. docs: methodology audit and training guide

```powershell
git add METHODOLOGY_AUDIT.md METHODOLOGY_TO_ADD.md README.md PROGRESS.md
git commit -m "docs: methodology audit and training guide

METHODOLOGY_AUDIT.md records what reading the preprocessing code
against the methodology turned up, tiered by severity, with each
finding marked as verified in code or inferred. Two findings were
corrected downward after measuring rather than left as written.

METHODOLOGY_TO_ADD.md is the plain-language version: what the
proposal does not yet say, organised by thesis section.

README gains a training section - the pipeline in order, why
run_training.py rather than the bare CLI, what to watch during a run,
and the disk and VRAM constraints."
```

---

## After

```powershell
git log --oneline -13
git status
```

`git status` should be clean apart from ignored directories.

---

## A note on attribution

I wrote or substantially shaped most of the code in commits 4-11. If
you want that recorded, append this to those commit messages:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Your call, and worth a thought for a thesis repo — some institutions
want AI assistance disclosed, some want it in the acknowledgements
rather than the git history, some do not care. I have left it off the
messages above rather than deciding for you.
