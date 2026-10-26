# PREPROCESSING - ElevenLabs

This folder contains all Python scripts for creating the ElevenLabs portion of the study's synthetic corpus.

These scripts use the following files as reference

- `manifests/elevenlabs_selected_speakers.csv` - PREREQUISITE; obtained from the 50/50 speaker split between FreeVC and ElevenLabs
- `transcripts/` - PREREQUISITE; the source of truth for the statements ElevenLabs turns into speech
- `data/processed/bonafide/` - PREREQUISITE; the 16 kHz mono bonafide audio, from `resample_dataset.py`
- `elevenlabs-reference/<speaker_id>/` - obtained from `select_reference_clips.py`, serves as reference audio for voice cloning
- `manifests/elevenlabs_voice_ids.txt` - obtained from `clone_voice.py`, maps each speaker ID to its ElevenLabs Voice ID
- `ELEVENLABS_API_KEY` - PREREQUISITE; set in `.env` at the project root

## IMPORTANT SCRIPTS

**`select_reference_clips.py`**

- scans each selected speaker's bonafide folder and randomly picks 3-5 utterances at least `--min-duration` seconds long
- copies them into `elevenlabs-reference/<speaker_id>/` to serve as the cloning reference
- these clips are excluded from generation, since a clip cannot be its own spoof

**`clone_voice.py`**

- creates an ElevenLabs Instant Voice Clone (IVC) per speaker and records the Voice ID
- can be done in batches of 30; the batch number decides WHICH speakers belong to the batch, while the existing voice-ID log decides whether a speaker still needs processing
- a speaker that already has a Voice ID is skipped safely, so re-running is harmless
- the selected-speakers CSV is never modified

**`generate_elevenlabs.py`**

- generates the synthetic speech, one clip per transcript line, using each speaker's cloned voice
- writes to `data/processed/elevenlabs` with the `.2.wav` suffix that marks an ElevenLabs spoof

## UTILITY SCRIPTS

**`delete_voices.py`**

- deletes ONLY ElevenLabs voices whose names start with `ceb-`
- this script exists to circumvent the 30 voice limit set for the subscription purchased by the authors of this study

**`regenerate_elevenlabs.py`**

- regenerates only the utterances listed in a .tsv file
- this script exists to help regenerate audio clips that are either faulty or affected by methodology adjustments

## RECOMMENDED PIPELINE

1. Run `select_reference_clips.py` to pick the reference clips for each speaker
2. Run `clone_voice.py` to create a Voice ID for each speaker
3. Run `generate_elevenlabs.py` to generate the audio clips

Use `delete_voices.py` to free up voice slots between batches, and `regenerate_elevenlabs.py` if any clips need to be regenerated.

Run `tests/check_missing_elevenlabs.py` afterwards to catch failed generations.
