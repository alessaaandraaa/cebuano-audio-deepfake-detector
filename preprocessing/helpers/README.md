# HELPER SCRIPTS

This folder contains Python helper scripts for auditing the corpus before it is used. Neither of them is part of the generation pipeline, and neither modifies audio unless explicitly told to.

**`audit_sources.py`**

- reports every distinct transcript source in the corpus, with clip count, speaker count and total duration, so you can see what the dataset is actually made of before cutting anything
- runs in AUDIT MODE by default; nothing is moved, deleted or modified unless `--apply` is passed
- with `--apply`, matching clips are MOVED into an `_excluded/` folder rather than deleted, so any cut is reversible
- pass `--spoof-root` for each spoof set when cutting, so the matching spoof clips go with their bonafide source; removing a source from bonafide only would leave orphaned spoofs and a detector can exploit that asymmetry
- `--processed-root` cross-checks whether an already-excluded source survived into the processed set

**`find_nonrandom_numbers.py`**

- finds utterances with a digit or number OUTSIDE the "Random Digit" source entries, i.e. numbers inside ordinary sentences
- this matters because the per-utterance 1-9 pronunciation lookup only applies to Random Digit entries; a bare digit in an ordinary sentence is handled by the fixed mapping instead
- reports how many such utterances exist and what they look like, so you can decide whether it matters for the corpus
- limited to the speakers listed in a CSV with a `speaker_id` column
