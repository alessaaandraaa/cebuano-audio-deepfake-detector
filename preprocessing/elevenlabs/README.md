
# PREPROCESSING - ElevenLabs

This folder contains all Python scripts for creating the ElevenLabs portion of the study's synthetic corpus. 

These scripts use the following files as reference
-  `manifests/elevenlabs_selected_speakers.csv` - PREQUISITE; obtained from the 50/50 speaker split between FreeVC and ElevenLabs
- `transcripts` - PREREQUISITE; used as the source of truth for the statements that ElevenLabs will transform to speech.
-  `elevenlabs-reference/<speaker_id>` - obtained from `select_reference_clips.py`, serves as reference audio for voice cloning
- `manifests/elevenlabs_voice_ids.txt` - obtained from `clone_voice.py`, contains the ElevenLabs Voice ID for each speaker. Mapped to speaker ID.



## IMPORTANT SCRIPTS
**`clone_voice.py`**
- creates a Voice ID 
- can be done in batches

**`generate_elevenlabs.py`**

**`select_reference_clips.py`**

## UTILITY SCRIPTS
**`delete_voices.py`**
- deletes ElevenLabs voices
- this script exists to circumvent the 30 voice limit set for the subscription purchased by the authors of this study.

**`regenerate_elevenlabs.py`**
- regenerates specific utterances as seen in a .tsv file
- this script exists to help regenerate audio clips that are either faulty or affected by methdology adjustments.


## RECOMMENDED PIPELINE
1. Run `select_reference_clips.py` for selecting reference clips for each speaker
2. Run `clone_voice.py` to create a Voice ID for each speaker
3. Run `generate_elevenlabs.py` to generate the audio clips

Use `delete_voices.py` to delete voice IDs, and `regenerate_elevenlabs.py` if any audio files need to be regenerated.