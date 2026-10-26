# CEBUANO AUDIO DEEPFAKE DETECTOR

Audio deepfake detection for Cebuano speech under real-world transmission conditions.

This repository contains everything used to build the corpus, train the detector and produce the reported results: the preprocessing and synthesis pipeline, the dataset construction, the shortcut analysis, the DeepFense training wrapper and the experiment configurations.

The detector is XLS-R 300M + AASIST with OC-Softmax. EER is the primary metric.

## THE CORPUS

A paired Cebuano corpus built from the Philippine Languages Database CEB set.

- 112,572 clips, 100.39 hours total
- 140 speakers, 81 female and 59 male
- bonafide 53.90 h, ElevenLabs 18.50 h, FreeVC 27.99 h
- every spoof clip has a bonafide clip with the same text, so the detector cannot learn content instead of synthesis artefacts
- split 70/15/15, speaker-disjoint and gender-stratified
  The corpus itself is not redistributed with this repository. What is here is the pipeline that builds it from the original PLD CEB corpus.

An earlier arm of the study used Meta MMS instead of FreeVC. Those scripts, configs and manifests are still here, since two of the runs in the results ledger are MMS runs. Anything named `meta-mms` belongs to that arm; anything named `voice-conv` or `_vc` belongs to the FreeVC arm, which is the one behind the reported results.

## WHAT IS IN THIS REPOSITORY

Every folder has its own README with the scripts it holds and the order to run them. Start there for detail; this file is the spine.

**`/preprocessing`** - builds the corpus: transcripts, speaker selection, ElevenLabs and FreeVC generation, resampling and normalisation

**`/manifests`** - speaker lists, generation records and every train/val/test split

**`/training`** - manifest building, the DeepFense wrapper, codec augmentation, and all scoring scripts

**`/tests`** - inspection and sanity-check scripts; none of them modify anything

**`/sampling`** - draws the stratified sample used for expert linguistic review of the spoof clips

**`/transcripts`**, **`/transcripts_mms`** - one .txt per speaker, the ground-truth text for synthesis

**`/elevenlabs-reference`** - the reference clips used for voice cloning

**`/outputs_archive`** - the exact config.yaml each reported checkpoint was trained with

`RESULTS_LEDGER.md` and `results_ledger.json` hold every reported metric: EER with bootstrap confidence intervals, AUC, and calibrated accuracy, precision, recall and F1 for all 50 cells.

## SETUP

Copy `.env.example` to `.env` and fill it in:

- `ELEVENLABS_API_KEY` - only needed to regenerate the ElevenLabs spoofs
- `PLD_CORPUS_ROOT` - absolute path to the ORIGINAL PLD CEB corpus, the folder holding one directory per speaker
  Scripts that read the original corpus load `.env` themselves, so `--root` is optional on all of them. `.env` is read by Python, not by your shell, so do NOT type `"$env:PLD_CORPUS_ROOT"` on the command line.

See ENVIRONMENT below for Python, FFmpeg and GPU setup.

## PART A - BUILD THE CORPUS

Run in order. Each step depends on the one before it. Full detail is in `preprocessing/README.md`.

1. **Inventory the bonafide audio.** `python preprocessing\inventory_bonafide.py --out manifests\manifest_bonafide.csv`
   Produces the speaker and gender manifest that everything downstream reads.
2. **Extract the transcripts.** `python preprocessing\extract_transcripts.py --out transcripts`
   One .txt per speaker, with the corpus-level cleanup that establishes the ground-truth vocabulary.
3. **Resample the bonafide audio to 16 kHz mono.** `python preprocessing\resample_dataset.py --out data\processed\bonafide`
   Both generation pipelines read `data\processed\bonafide`, so this must run before either of them. Originals are never modified.
4. **Split the speakers 50/50.** `python preprocessing\select_speakers.py --manifest manifests\manifest_bonafide.csv --fraction 0.5 --seed 42 --out manifests\vc_selected_speakers.csv`
   Half the speakers go to ElevenLabs, half to FreeVC. Gender-stratified, fixed seed.
5. **Generate the spoofs.** ElevenLabs and FreeVC are separate pipelines with their own READMEs: `preprocessing/elevenlabs/` and `preprocessing/freevc/`.
6. **Normalise the corpus.** See PART B step 1 - normalisation is the bridge between the two halves.

## PART B - TRAIN AND EVALUATE

Run in order. Full detail is in `training/README.md`.

### 1. Normalise the corpus

Puts every clip, both classes, through one identical chain: decode to mono float, resample to 16 kHz with one resampler, trim silence at a fixed relative threshold, peak-normalise to 1.0, write PCM_16. This is the ASVspoof 5 shortcut-artefact operation, and it exists so the class label stops being readable from production artefacts rather than from speech.

```powershell
python preprocessing\normalise_corpus.py `
  --bonafide-root data\processed\bonafide `
  --spoof-root voice-conv=data\processed\voice-conv `
  --spoof-root elevenlabs=data\processed\elevenlabs `
  --out-root data\normalised `
  --apply `
  --workers 6
```

Leave off `--apply` for a dry run, which processes a sample in memory and reports before/after without writing. Do that first. The real run is not resumable and takes hours on the full corpus. Originals are never touched; output goes to a parallel tree so the before/after probe comparison stays reproducible.

### 2. Build the manifests

```powershell
python training\build_manifests.py `
  --bonafide-root data\normalised\bonafide `
  --spoof-root voice-conv=data\normalised\voice-conv `
  --spoof-root elevenlabs=data\normalised\elevenlabs `
  --speaker-manifest manifests\manifest_bonafide.csv `
  --out manifests\splits_vc `
  --unpaired-report manifests\splits_vc\unpaired_excluded.csv
```

Then open `manifests\splits_vc\split_summary.txt` and confirm it says `RESULT: speaker-disjoint`. Do not continue if it does not.

Manifests contain ABSOLUTE paths, so build them on the machine that will train. Copying manifests between machines does not work.

### 3. Check the pairing guard

The guard drops any spoof clip whose bonafide counterpart is missing, so the 1:1 pairing holds and the detector cannot learn content differences. It reports what it dropped as part of step 2.

For the MMS corpus the expected result was 367 of 56,653 spoof clips excluded (0.65%) - 357 meta-mms and 10 elevenlabs - leaving 56,286 bonafide and 56,286 spoof.

If your number differs from what the summary records for your corpus, investigate upstream before training. `--max-unpaired-frac` (default 0.05) is a safety stop, not a tuning knob. `--allow-unpaired` overrides it and should essentially never be used.

### 4. Run the shortcut probe

Measures how much of the task can be solved from acoustic and file-level features alone, without listening to the speech. The number to compare against is the trained detector's EER: if the probe is close to it, the detector may be reading production artefacts.

```powershell
python training\helpers\probe_shortcuts.py `
  --splits-dir manifests\splits_vc `
  --eval-split test `
  --workers 4 `
  --cache outputs\probe_test.parquet
```

Run it before and after normalisation and compare. On this corpus, normalisation closed `peak_db`, `rms_db`, `crest_db`, `hf_ratio_7k` and `duration_s`, but opened `trail_silence_s` and `dc_offset`. Report what remains as a limitation rather than normalising until it disappears - over-normalising destroys the vocoder artefacts the detector needs.

### 5. Verify the DeepFense patches

```powershell
python training\helpers\deepfense_patches.py --verify
```

All three checks must pass. The patches fix DeepFense bugs that only bite on Windows, where DataLoader workers are spawned rather than forked, and they enable gradient clipping. Do not skip this.

### 6. Check FFmpeg has both codecs

```powershell
python training\codec_opus_amr.py --check-ffmpeg
python training\codec_opus_amr.py --self-test
```

You need a build with `libopus` and `libopencore_amrnb`. Training will fail partway through without them.

### 7. Train

```powershell
python training\run_training.py --config training\configs_vc\train_vc.yaml
```

Always launch through `run_training.py`, never `deepfense train` directly. The wrapper is what imports `codec_opus_amr.py`, and that import is what registers the `opus_codec` and `amrnb_codec` transforms. The bare CLI never imports it, so a config referencing those transforms fails with a registry KeyError.

Early in the log, confirm both of these appear:

```text
Applied deepfense patches: ...
Gradient clipping enabled: max_grad_norm=5.0
```

If gradient clipping is missing, stop and investigate.

### 8. Score the results

A single checkpoint is evaluated against several test conditions without retraining - clean, Opus and AMR are separate test configs, not separate runs.

```powershell
python training\make_per_source_eval.py --splits-dir manifests\splits_vc --apply
python training\score_predictions.py --calibrate <val-predictions> --evaluate <test-predictions>
python training\per_source_eer.py --predictions <predictions> --splits-dir manifests\splits_vc
python training\speaker_bootstrap.py --pred <predictions> --manifest manifests\splits_vc\test.parquet
```

Use `speaker_bootstrap.py` for confidence intervals and for paired differences between two runs. It resamples SPEAKERS, not clips, because clips from one speaker score alike and clip-level intervals come out too narrow.

Do not report raw `ACC` as the headline number. OC-Softmax's default threshold is not meaningful; use EER, and use `score_predictions.py` for threshold-dependent metrics at a threshold calibrated on a held-out split.

## TRAINING CONFIGURATIONS

`/configs` holds the MMS arm, `/configs_vc` the FreeVC arm. The FreeVC ones are behind the reported results.

Training configs in `/configs_vc`:

| Config                                         | Purpose                           |
| ---------------------------------------------- | --------------------------------- |
| `train_vc.yaml`                                | main run, Opus + AMR augmentation |
| `noaug.yaml`                                   | no codec augmentation             |
| `train_vc_opus_only.yaml`                      | Opus augmentation only            |
| `train_vc_amr_only.yaml`                       | AMR augmentation only             |
| `train_vc_augval.yaml`                         | augmented validation variant      |
| `loso_elevenlabs.yaml`, `loso_voice-conv.yaml` | leave-one-system-out              |

The `test_*.yaml` files in the same folder are evaluation configs, not training runs. They score an existing checkpoint against one condition - clean, Opus, AMR, a held-out system, or a validation split.

Scaffolding configs live in `/configs` alongside the MMS arm: `smoke.yaml` (1 epoch, frozen frontend), `smoke_unfrozen.yaml` (1 epoch, checks joint fine-tuning fits in GPU memory) and `timing_2epoch.yaml` (timing test).

The timing config keeps `T_max: 30` even though it runs fewer epochs, so its learning-rate schedule matches the first part of a full run.

## EXPERIMENT MATRIX

```powershell
python training\make_loso_splits.py --splits-dir manifests\splits_vc --out manifests\loso_vc
python training\make_ablation_configs.py --apply
```

The ablation configs are GENERATED from the main config rather than hand-copied, so the arms are guaranteed to differ in exactly the variable under test.

| Run                  | Purpose                                       |
| -------------------- | --------------------------------------------- |
| main                 | Opus + AMR augmentation                       |
| noaug                | effect of codec augmentation                  |
| opus-only / amr-only | which codec carries the effect                |
| loso\_`<system>`     | leave-one-synthesis-system-out generalisation |

Run experiments one at a time on a single GPU.

## WHAT TO MONITOR

**Validation EER** - from the first epoch. Compare it against the shortcut-probe floor, not against zero.

**Gradient clipping** - watch for `[grad-clip]`. Frequent clipping suggests the learning rate is aggressive for this dataset.

**Overfitting** - training loss falling while validation loss rises. Early stopping is configured, but watch it anyway.

**Accuracy** - do not use raw `ACC` to judge the run. See step 8.

## BEFORE A LONG RUN

**Storage.** Each checkpoint is about 3.8 GB, and a 30-epoch run produces roughly 40-60 GB depending on how many are kept. Check with `df -h` and `du -sh outputs/`. Keep `train.log` and `results.json` for every experiment that matters - they are small and they are the record.

**GPU.** The tested setup is batch size 4 with gradient accumulation 4, an effective batch of 16. Batch size 8 caused `torch.OutOfMemoryError` on 8 GB VRAM. Do not run games or other GPU jobs during training; two concurrent runs will not fit.

**Persistent session.** On a remote Linux box, never run training in a bare SSH session. Use `tmux new -s deepfense`, start the run, detach with Ctrl+B then D, and reattach later with `tmux attach -t deepfense`.

**Checklist.**

```text
[ ] correct repository and dataset on the training machine
[ ] bonafide, ElevenLabs and FreeVC sets all present
[ ] manifests built ON this machine
[ ] split_summary.txt says speaker-disjoint
[ ] pairing guard count matches what the corpus expects
[ ] shortcut probe run and interpreted
[ ] ffmpeg has libopus and libopencore_amrnb
[ ] correct conda/venv active, PyTorch sees CUDA
[ ] deepfense patches verified
[ ] enough disk space, no competing GPU jobs
[ ] persistent session configured
```

## VERSIONS

The exact environment that produced the reported results. `requirements-lock.txt` holds all 139 resolved packages; these are the ones a reviewer will ask about.

**Python 3.11.9** on Windows, **CUDA 12.8**.

| Component | Version | Notes |
| --- | --- | --- |
| `deepfense` | 0.2.2 | training framework; patched at runtime, see `training/helpers/deepfense_patches.py` |
| `torch` | 2.11.0+cu128 | |
| `torchaudio` | 2.11.0+cu128 | |
| `torchcodec` | 0.16.0 | installed `--no-deps`, see below |
| `transformers` | 4.55.4 | loads the XLS-R frontend |
| `coqui-tts` | 0.27.5 | provides FreeVC |
| `elevenlabs` | 2.65.0 | |
| `numpy` | 2.1.3 | |
| `soundfile` | 0.14.0 | |
| `librosa` | 0.11.0 | |
| `pandas` | 2.3.3 | |
| `pyarrow` | 25.0.1 | Parquet manifests |
| `scikit-learn` | 1.7.2 | |

**Models.**

| Model | Identifier |
| --- | --- |
| Frontend | `facebook/wav2vec2-xls-r-300m` |
| Voice conversion | `voice_conversion_models/multilingual/vctk/freevc24`, via coqui-tts |
| Text to speech | ElevenLabs Instant Voice Clone, hosted API |

FreeVC is not vendored. coqui-tts downloads the `freevc24` checkpoint on first run. ElevenLabs is a hosted service, so its model version is whatever the API served at generation time -- the spoofs were generated 26-28 August 2026.

**Two install traps.**

`torchcodec` must be installed with `--no-deps`:

```powershell
venv\Scripts\python.exe -m pip install --no-deps torchcodec
```

From torch 2.9 onward, `TTS/__init__.py` checks that `torchcodec` merely EXISTS via `importlib.util.find_spec`. FreeVC never calls it; it decodes with librosa, so satisfying the check is enough. A plain `pip install coqui-tts[codec]` re-resolves the entire coqui-tts dependency tree inside the venv you train in, which is how a working torch/deepfense pair gets quietly replaced.

**FFmpeg** must be built with `libopus` and `libopencore_amrnb`. Record the version you used, since the codec augmentation depends on it:

```powershell
python training\codec_opus_amr.py --check-ffmpeg
ffmpeg -version
```

## ENVIRONMENT

### Windows

```powershell
py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements-lock.txt
pip install --no-deps torchcodec
```

`torchcodec` goes in separately and with `--no-deps` -- see VERSIONS above for why.

Install an FFmpeg build containing `libopus` and `libopencore_amrnb`, and put it on PATH.

### Linux / Conda

Create the `deepfense` environment, then activate it before every session. Keep the code and environment on fast local storage and the dataset on whatever bulk storage the machine provides.

Commands are the same as the PowerShell ones above with forward slashes and `\` continuations instead of backticks. One shell gotcha: `~` does not expand after `=`, so write

```bash
--spoof-root "voice-conv=$HOME/archive/normalised/voice-conv"
```

and not `--spoof-root voice-conv=~/archive/normalised/voice-conv`.

### FFmpeg without admin rights

Install it under your home directory, then put it on PATH:

```bash
export PATH="$HOME/bin:$PATH"
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
which ffmpeg
```

## SVC1 (UNIVERSITY SERVER)

Specific to SVC1. Do not assume any of it applies to another Linux machine.

**SSH.** Custom port: `ssh -p 22081 <your-username>@svc1.dcism.org`

**Storage.** The home directory is on NVMe; `~/archive` is a symlink to personal HDD storage at `/disk2/storage/users/<username>`. Check both with `df -h ~ ~/archive`. `/disk2` itself is not writable - always go through `~/archive`.

A practical layout keeps code, conda and user-local binaries on the fast home storage and the large normalised dataset under `~/archive/normalised/`.

**Before training.** Run `nvidia-smi` and check nobody else's job is already on the GPU. Do not kill other users' processes.

**After reconnecting.** `conda activate deepfense`, `cd` into the repository, then confirm `python --version`, `which ffmpeg`, `nvidia-smi`, `python training/codec_opus_amr.py --check-ffmpeg` and `python training/helpers/deepfense_patches.py --verify`.

## TROUBLESHOOTING

**`ffmpeg: command not found`** - see "FFmpeg without admin rights" above.

**`AMR-NB encoder 'libopencore_amrnb': MISSING`** - the FFmpeg build lacks the encoder. Check with `ffmpeg -encoders | grep -i amr` and `ffmpeg -buildconf | grep -i opencore`. Fix it before the main run; training will fail partway through otherwise.

**Training fails with an AMR transform error** - same cause. Run `python training/codec_opus_amr.py --check-ffmpeg`.

**KeyError from the Transform registry** - you launched `deepfense train` directly. Use `run_training.py`.

**`conda-forge` cannot be accessed** - some institutional networks block it. Use whatever package source your institution provides, or install user-local.

**`CUDA: False`** - check `nvidia-smi`, then `python -c "import torch; print(torch.cuda.is_available())"`. If the GPU is visible to one and not the other, the installed PyTorch build does not match the machine's CUDA environment.

**Manifest paths are wrong** - manifests store absolute paths. Rebuild them on the machine that will train. Do not copy them across machines.

**`expected one argument` after `--root`** - you typed `"$env:PLD_CORPUS_ROOT"` on the command line. PowerShell does not read `.env`, so it expanded to an empty string. Omit `--root` entirely; the script reads `.env` itself.

**SSH disconnects during training** - see "Persistent session" above.

## METHODOLOGY NOTES

**Paired design.** Every spoof utterance corresponds to a bonafide utterance with the same text, so the detector cannot separate the classes on linguistic content. The pairing guard enforces this by dropping unpaired spoof clips.

**Speaker-disjoint evaluation.** Splits are partitioned by speaker, not by clip, so no speaker appears in more than one split and speaker identity cannot leak. Always confirm `RESULT: speaker-disjoint` in the split summary.

**Shortcut risk.** Both classes came off different production chains, and that difference is learnable without listening to the speech. The probe measures it; identical preprocessing across both classes is what reduces it. A residual leak is not automatically fatal, but it has to be measured and reported rather than assumed away.

**Normalisation is peak, and only peak.** ASVspoof 5's shortcut-artefact pipeline scales each waveform so its peak equals 1.0, and that is what this pipeline does. Gated BS.1770-4 loudness, plain RMS, a common dither floor and an MP3 equalisation pass were all implemented and all removed: none is what this field does, and each would mean defending a departure from the literature on our own measurements. The dither also backfired when measured - at -75 dBFS it sat under half a least-significant bit, quiet passages still rounded to bit-exact zero, and `frac_exact_zero` went from 0.5000 to 0.3115. It created the cue it was meant to remove.

**Codec augmentation.** Opus at a stochastic 6-32 kbps and AMR-NB cycling its eight standard rates, both applied on the fly through ffmpeg so nothing extra is written to disk. AMR-NB is an 8 kHz codec, so the transform resamples down, encodes, decodes and resamples back - that round trip is part of the channel degradation being simulated, not a bug.
