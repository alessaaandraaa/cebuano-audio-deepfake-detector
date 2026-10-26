# SAMPLING

This folder contains the script that draws the verification sample used for expert linguistic review of the synthetic half of the dataset.

The point is that a human reviewer listens to a sample of the spoof clips and judges whether they are intelligible Cebuano, rather than the study asserting it.

These scripts use the following files as reference

- `data/processed/elevenlabs/` and `data/processed/meta-mms/` - PREREQUISITE; the generated spoof audio
- `transcripts/` - PREREQUISITE; used to print what each sampled clip was supposed to say

**`select_verification_samples.py`**

- draws 50 ElevenLabs clips and 50 Meta MMS clips, 100 in total
- stratified by duration into short (< 2.0s), medium (2.0-5.0s) and long (> 5.0s), at 25% / 50% / 25%
- stratified rather than drawn at random so that short utterances, where synthesis is weakest, are represented instead of being washed out by the corpus-wide duration distribution
- if a bin has fewer files than its quota, the unused quota is redistributed to the bins that still have capacity
- samples are shuffled before numbering, so `ceb_sample_001` to `100` are not grouped by source and the reviewer cannot tell which system produced which clip
- source audio is COPIED, never moved or modified

Output goes to `data/samples/`, with a `samples.tsv` recording the source, speaker, original filename, duration, duration bin and transcript for every sampled clip.

## IMPORTANT

Pass `--seed` and record the value used. The sample is the basis for the validation results, so the draw has to be reproducible.
