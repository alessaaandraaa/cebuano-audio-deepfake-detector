# PREPROCESSING - FreeVC

This folder contains all Python scripts for creating the FreeVC portion of the study's synthetic corpus.

These scripts use the following files as reference

- `manifests/vc_selected_speakers.csv` - PREREQUISITE; obtained from the 50/50 speaker split between FreeVC and ElevenLabs
- `data/processed/bonafide/` - PREREQUISITE; the 16 kHz mono bonafide audio, from `resample_dataset.py`
- `manifests/vc_speaker_pairs.csv` - obtained from `make_vc_pairs.py`, the A -> B pairing

## IMPORTANT SCRIPTS

**`make_vc_pairs.py`**

- assigns each source speaker a TARGET VOICE, so A's utterances end up spoken in B's voice
- proves the assignment is safe before a single clip is generated: no speaker is ever paired with itself, and the pairing is a derangement
- writes `manifests/vc_speaker_pairs.csv`

**`vc_convert.py`**

- uses the pairing to perform the voice conversion, writing to `data/processed/voice-conv`
- this file was previously a script for testing three different VC models; the final study uses FreeVC only
- also writes `manifests/vc_reference_clips.csv` and `manifests/vc_generation_report.csv`, which record the reference clip chosen and the per-clip outcome

## RECOMMENDED PIPELINE

1. Run `make_vc_pairs.py` to pair speakers and verify the pairing
2. Run `vc_convert.py` to create the voice-converted audio clips

Run it from the project root, not from inside this folder, so the `preprocessing.freevc.*` imports resolve.
