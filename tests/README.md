# TEST SCRIPTS

This folder contains all Python scripts for inspecting and sanity-checking the corpus. None of them modify audio, transcripts or manifests.

**`check_durations.py`**

- reports duration stats per speaker folder and overall, reading the audio files directly
- no logs and no manifests involved

**`count_durations.py`**

- quick duration summary straight from the JSONL manifest, without opening any audio
- reports total clips, how many fall under 1 second, and the mean

**`check_leading_silence.py`**

- reports leading silence for every audio file under a root
- run it on the bonafide root and a spoof root separately, then compare; a gap between the two is a shortcut the detector can learn instead of learning speech

**`check_missing_elevenlabs.py`**

- cross-references the ElevenLabs output against the transcripts to find failed or missing generations
- accounts for the `.2.wav` suffix and ignores the clips that were skipped on purpose because they are reference clips

**`count_characters.py`**

- counts the characters in the utterances of every .log file under a root
- metadata lines such as SpeakerID and SpeakerGender are ignored

**`determine_number_logs.py`**

- finds every "Random Digit" entry in the corpus, detects the number and translates it with `preprocessing/number_mapping/numbers.json`
- writes `manifests/transcripts_regenerate.txt`
- takes no arguments; the corpus path comes from `PLD_CORPUS_ROOT` in `.env`
