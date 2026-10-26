# PREPROCESSING SCRIPTS

This folder contains all scripts for the study's preprocessing pipeline: everything that turns the original PLD CEB corpus into the paired bonafide/spoof dataset the detector trains on.

These scripts use the following file as reference

- `PLD_CORPUS_ROOT` - PREREQUISITE; set in `.env` at the project root (copy `.env.example`). Points at the ORIGINAL corpus, the folder holding one directory per speaker. The scripts read `.env` themselves, so `--root` is optional on all of them.

## FOLDERS

**`/elevenlabs`** - scripts for ElevenLabs voice cloning and speech generation

**`/freevc`** - scripts for FreeVC voice conversion

**`/helpers`** - helper scripts for auditing the corpus before it is used

**`/number_mapping`** - the digit -> word number mapping and the script that verifies it

## IMPORTANT SCRIPTS

**`inventory_bonafide.py`**

- scans the original corpus and writes one row per included bonafide clip, with speaker ID and gender
- uses each speaker's .log file to decide which clips belong in the dataset
- TGL_* transcript sources are excluded automatically
- produces `manifests/manifest_bonafide.csv`, the source of truth for speaker gender everywhere downstream

**`extract_transcripts.py`**

- writes one .txt file per speaker from that speaker's .log file
- performs the corpus-level cleanup that establishes the ground-truth vocabulary: removes English glosses, drops unwanted sources, expands digits, collapses stray whitespace
- digits 1-9 are looked up PER UTTERANCE from `manifests/original_pronunciations.tsv`, since they have no fixed Cebuano form across speakers; 10-100 use the fixed mapping in `/number_mapping`

**`resample_dataset.py`**

- converts the raw bonafide audio to 16 kHz mono and writes it to `data/processed/bonafide`
- this is what both generation pipelines read, so it has to run before either of them
- raw originals are never modified

**`select_speakers.py`**

- splits the speakers 50/50 between ElevenLabs and FreeVC, stratified by gender so each half keeps the corpus ratio
- fixed seed, so the same manifest and seed always give the same selection

**`normalise_corpus.py`**

- puts every clip, both classes, through one identical chain: decode to mono float, resample to 16 kHz with one resampler, trim silence at a fixed relative threshold, peak-normalise to 1.0, write PCM_16
- this is the ASVspoof 5 shortcut-artefact operation; the point is that the class label stops being readable from production artefacts rather than from speech
- run it WITHOUT `--apply` first for a dry run that reports before/after and writes nothing
- originals are never modified; output goes to a parallel tree so the before/after probe comparison stays reproducible

## RECOMMENDED PIPELINE

1. Run `inventory_bonafide.py` to build the speaker and gender manifest
2. Run `extract_transcripts.py` to produce the per-speaker transcripts
3. Run `resample_dataset.py` to get the bonafide audio to 16 kHz mono
4. Run `select_speakers.py` to split the speakers between the two synthesis systems
5. Generate the spoofs: see `/elevenlabs` and `/freevc`, which have their own pipelines
6. Run `normalise_corpus.py` over bonafide and both spoof sets together

After that, move to `training/` to build the manifests and train. The root `README.md` has the full pipeline end to end.

Use `/helpers` to audit what is in the corpus before committing to any of it.
