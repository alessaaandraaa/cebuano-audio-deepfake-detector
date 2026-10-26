# MANIFESTS

This folder contains all manifests, speaker lists and split files used by the pipeline. These are the files that record which clip belongs to which speaker, which speaker belongs to which split, and which synthesis system produced what.

## SPEAKER LISTS

**`manifest_bonafide.csv`**

- one row per included bonafide clip, with speaker ID and gender
- obtained from `preprocessing/inventory_bonafide.py`
- source of truth for speaker gender in every split built downstream

**`elevenlabs_selected_speakers.csv`** and **`vc_selected_speakers.csv`**

- the 50/50 speaker split between ElevenLabs and FreeVC
- obtained from `preprocessing/select_speakers.py`
- PREREQUISITE for both generation pipelines

**`elevenlabs_voice_ids.txt`**

- maps each speaker ID to its ElevenLabs Voice ID
- obtained from `preprocessing/elevenlabs/clone_voice.py`
- not tracked by git, since the IDs belong to the subscription purchased by the authors of this study

**`vc_speaker_pairs.csv`**

- the A -> B pairs for voice conversion
- obtained from `preprocessing/freevc/make_vc_pairs.py`

## GENERATION RECORDS

**`original_pronunciations.tsv`**

- per-utterance pronunciation for the digits 1-9, which have no fixed Cebuano form across speakers
- read by `preprocessing/extract_transcripts.py`, which looks each utterance up by wav filename so every recording keeps the pronunciation actually spoken in it

**`vc_reference_clips.csv`** and **`vc_generation_report.csv`**

- reference clip choice and per-clip outcome for the FreeVC run
- obtained from `preprocessing/freevc/vc_convert.py`

## SPLITS

Every split folder holds `train.parquet`, `val.parquet`, `test.parquet` and a `split_summary.txt`, all from `training/build_manifests.py`. The split is 70/15/15, speaker-disjoint and stratified by gender.

**`/splits`** - first build, Meta MMS corpus

**`/splits_norm`** - Meta MMS corpus after normalisation; what the MMS runs read

**`/splits_vc`** - FreeVC corpus after normalisation; what the reported results read

**`/splits_smoke`** - small subset for smoke tests only

The per-source test files (`test_elevenlabs.parquet`, `test_meta-mms.parquet`, `test_voice-conv.parquet`) and `unpaired_excluded.csv` come from `training/make_per_source_eval.py`.

## LEAVE-ONE-SYSTEM-OUT

**`/loso`** and **`/loso_vc`**

- obtained from `training/make_loso_splits.py`
- one subfolder per held-out synthesis system
- each holds `train.parquet`, `val.parquet`, `test_matched.parquet` and `test_unseen.parquet`
- `test_matched` is the system the model trained on, `test_unseen` is the one it never saw; the gap between the two EERs is the generalisation result

## OTHER FILES

**`audio_summary.txt`** and **`missing_number_files.txt`**

- notes kept from earlier corpus checks
- nothing in the pipeline reads them
