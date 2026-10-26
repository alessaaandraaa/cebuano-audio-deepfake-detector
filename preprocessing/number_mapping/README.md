# PREPROCESSING - NUMBER MAPPING

This folder contains the digit -> word number mapping and the script that verifies it.

Numbers have to be spelled out before synthesis, because a TTS system handed a bare digit will not say it the way the speaker did.

**`numbers.json`**

- number mapping for digits 10-100
- a fixed corpus-wide mapping, applied regardless of source: none of the audio clips had different utterances for these digits
- read by `preprocessing/extract_transcripts.py`

**`check_random_digits.py`**

- helper script that verifies whether the digits 1-9 were all successfully mapped to their respective word
- scans the ORIGINAL corpus for Random Digit utterances belonging to speakers that have ElevenLabs voice IDs
- writes a TSV of `speaker_id / wav_filename / number / tts_text`, which feeds `preprocessing/elevenlabs/regenerate_elevenlabs.py`
- the original corpus is never modified

Digits 1-9 are deliberately NOT in `numbers.json`. They have no fixed pronunciation across speakers -- "uno" vs "usa", "dos" vs "duha", "sais" vs "unom" -- so they are looked up per utterance from `manifests/original_pronunciations.tsv` instead, and each recording keeps the pronunciation actually spoken in it.
