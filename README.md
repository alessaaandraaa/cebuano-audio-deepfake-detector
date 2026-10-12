# Cebuano Audio Deepfake Detector

Audio deepfake detection for Cebuano speech under real-world transmission conditions.

This repository contains the preprocessing, dataset construction, shortcut analysis, DeepFense training pipeline, and experiment configuration used for the project.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)

  * [Linux](#linux)
  * [Windows](#windows)
* [Environment Setup](#environment-setup)

  * [Linux / Conda](#linux--conda-recommended)
  * [Windows / venv](#windows--venv)
* [FFmpeg Requirements](#ffmpeg-requirements)
* [Dataset Layout](#dataset-layout)
* [Training Pipeline](#training-pipeline)

  * [1. Normalize the Corpus](#1-normalize-the-corpus)
  * [2. Build Manifests](#2-build-manifests)
  * [3. Pairing Guard](#3-pairing-guard)
  * [4. Shortcut Probe](#4-shortcut-probe)
  * [5. DeepFense Preflight](#5-deepfense-preflight)
  * [6. Train](#6-train)
* [Training Configurations](#training-configurations)
* [Experiment Matrix](#experiment-matrix)
* [What to Monitor](#what-to-monitor)
* [Long-Running Training](#long-running-training)
* [Storage Requirements](#storage-requirements)
* [GPU Requirements](#gpu-requirements)
* [SVC1 Example Setup](#svc1-example-setup)
* [Troubleshooting](#troubleshooting)
* [Important Methodology Notes](#important-methodology-notes)

---

# Overview

The training pipeline is designed around a paired Cebuano speech corpus containing:

* **Bonafide** human speech
* **Meta MMS** synthetic speech
* **ElevenLabs** synthetic / cloned speech

The pipeline performs:

1. Audio normalization
2. Speaker-disjoint dataset splitting
3. Bonafide/spoof pairing
4. Shortcut analysis
5. DeepFense runtime preflight
6. Model training
7. Codec-based augmentation and evaluation

The intended dataset split is:

```text
70% train
15% validation
15% test
```

The split is **speaker-disjoint** and **gender-stratified**.

---

# Repository Structure

A simplified repository structure:

```text
cebuano-audio-deepfake-detector/
│
├── data/
│   └── ...
│
├── manifests/
│   ├── manifest_bonafide.csv
│   └── splits_norm/
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── split_summary.txt
│       └── unpaired_excluded.csv
│
├── training/
│   ├── run_training.py
│   ├── normalise_corpus.py
│   ├── build_manifests.py
│   ├── probe_shortcuts.py
│   ├── deepfense_patches.py
│   ├── codec_opus_amr.py
│   ├── make_loso_splits.py
│   ├── make_ablation_configs.py
│   └── configs/
│       ├── smoke.yaml
│       ├── smoke_unfrozen.yaml
│       ├── timing_2epoch.yaml
│       └── train_main.yaml
│
├── outputs/
│   └── ...
│
├── METHODOLOGY_AUDIT.md
└── README.md
```

Generated datasets and training outputs are intentionally kept outside version control where appropriate.

---

# Requirements

## Linux

Recommended for GPU servers and university supercomputers.

Tested environment:

```text
Linux
ARM64 / aarch64
Python 3.10
Conda
PyTorch with CUDA support
NVIDIA GB10
CUDA 13.0
```

The exact GPU and CUDA version may differ on another machine.

Check the machine:

```bash
uname -m
python3 --version
nvidia-smi
```

For NVIDIA GPUs, verify that PyTorch can see the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected result should indicate:

```text
CUDA: True
GPU: <your NVIDIA GPU>
```

---

## Windows

The original development environment uses:

```text
Python 3.11
venv
NVIDIA GPU
FFmpeg
```

Create the environment:

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

Always activate the environment before working on the project.

---

# Environment Setup

## Linux / Conda (Recommended)

Conda is recommended for Linux environments, particularly when Python versions and native dependencies need to be isolated.

Install Miniconda if it is not already available.

Then create the DeepFense environment:

```bash
conda create -n deepfense python=3.10
```

Activate it:

```bash
conda activate deepfense
```

Install DeepFense:

```bash
pip install deepfense
```

Install the project's Python dependencies:

```bash
pip install soundfile librosa pandas numpy
pip install wandb scikit-learn
pip install transformers accelerate scipy
pip install elevenlabs python-dotenv
```

Install PyTorch according to the CUDA version and architecture of the target machine.

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Important

Do not assume that a PyTorch build intended for x86_64 will work on ARM64.

Check the architecture first:

```bash
uname -m
```

For example:

```text
aarch64
```

means the machine is ARM64.

---

## Windows / venv

Create and activate the environment:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
```

Install PyTorch using the build appropriate for the installed NVIDIA driver/CUDA compatibility.

Then:

```powershell
pip install soundfile librosa pandas numpy
pip install wandb scikit-learn
pip install transformers accelerate scipy
pip install elevenlabs python-dotenv
```

The project also provides a Conda-based DeepFense environment:

```powershell
conda create -n deepfense python=3.10
conda activate deepfense
pip install deepfense
```

---

# FFmpeg Requirements

FFmpeg is required by the preprocessing and codec augmentation pipeline.

The FFmpeg build must contain:

* `libopus`
* `libopencore_amrnb`
* `libmp3lame`

The important requirement is that these codecs are **compiled into the FFmpeg binary**, not merely installed somewhere on the system.

Check the project-specific requirements:

```bash
python training/codec_opus_amr.py --check-ffmpeg
```

and:

```bash
python training/normalise_corpus.py --check-ffmpeg
```

A successful preflight should report:

```text
Opus     encoder 'libopus': AVAILABLE
AMR-NB   encoder 'libopencore_amrnb': AVAILABLE
```

---

## Linux: User-local FFmpeg

If you do not have administrator/root access, install FFmpeg into a directory inside your home directory.

For example:

```text
~/bin/ffmpeg
```

Then add it to your PATH:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
which ffmpeg
```

Then:

```bash
ffmpeg -encoders | grep -i amr
```

You should see:

```text
libopencore_amrnb
```

Also verify:

```bash
ffmpeg -buildconf | grep -i opencore
```

You should see an option similar to:

```text
--enable-libopencore_amrnb
```

### Important

Not every static FFmpeg build contains AMR-NB support.

If:

```bash
python training/codec_opus_amr.py --check-ffmpeg
```

reports:

```text
AMR-NB ... MISSING
```

training will fail when the AMR augmentation transform is initialized.

Do not start a long training run until both required codecs are available.

---

# Dataset Layout

The normalized dataset should contain three directories:

```text
normalised/
├── bonafide/
├── meta-mms/
└── elevenlabs/
```

For example:

```text
~/archive/normalised/
├── bonafide/
├── meta-mms/
└── elevenlabs/
```

The exact storage location is machine-dependent.

---

# Training Pipeline

Run the following steps **in order**.

```text
Dataset
   │
   ▼
1. Normalize corpus
   │
   ▼
2. Build manifests
   │
   ▼
3. Pairing guard
   │
   ▼
4. Shortcut probe
   │
   ▼
5. DeepFense preflight
   │
   ▼
6. Train
```

Each step depends on the previous one.

---

# 1. Normalize the Corpus

Normalization puts every audio clip through a consistent processing chain.

This is intended to reduce class-specific production artifacts such as differences in loudness, silence, sampling rate, and related properties.

See `METHODOLOGY_AUDIT.md` for the project's methodology discussion.

## Linux

```bash
python training/normalise_corpus.py \
  --bonafide-root data/processed/bonafide \
  --spoof-root meta-mms=data/processed/meta-mms \
  --spoof-root elevenlabs=data/processed/elevenlabs \
  --out-root data/normalised \
  --apply \
  --workers 6
```

## Windows

```powershell
python training\normalise_corpus.py `
  --bonafide-root data\processed\bonafide `
  --spoof-root meta-mms=data\processed\meta-mms `
  --spoof-root elevenlabs=data\processed\elevenlabs `
  --out-root data\normalised `
  --apply `
  --workers 6
```

### Dry run first

Run the command without:

```text
--apply
```

This performs the preprocessing as a dry run and reports before/after information.

The normalization process is not resumable.

For the full corpus, expect the process to take significant time.

---

# 2. Build Manifests

The manifest builder creates the train/validation/test datasets.

The split is:

```text
70% train
15% validation
15% test
```

with:

* speaker-disjoint splitting
* gender stratification
* paired bonafide/spoof data

## Linux

```bash
python training/build_manifests.py \
  --bonafide-root "$HOME/archive/normalised/bonafide" \
  --spoof-root "meta-mms=$HOME/archive/normalised/meta-mms" \
  --spoof-root "elevenlabs=$HOME/archive/normalised/elevenlabs" \
  --speaker-manifest manifests/manifest_bonafide.csv \
  --out manifests/splits_norm \
  --unpaired-report manifests/splits_norm/unpaired_excluded.csv
```

### Important Linux shell note

Do not write:

```bash
--spoof-root meta-mms=~/archive/normalised/meta-mms
```

The `~` will not necessarily expand correctly when it occurs after `=`.

Use:

```bash
--spoof-root "meta-mms=$HOME/archive/normalised/meta-mms"
```

instead.

## Windows

```powershell
python training\build_manifests.py `
  --bonafide-root data\normalised\bonafide `
  --spoof-root meta-mms=data\normalised\meta-mms `
  --spoof-root elevenlabs=data\normalised\elevenlabs `
  --speaker-manifest manifests\manifest_bonafide.csv `
  --out manifests\splits_norm `
  --unpaired-report manifests\splits_norm\unpaired_excluded.csv
```

### Verify the split

Read:

```text
manifests/splits_norm/split_summary.txt
```

The leakage check should report:

```text
RESULT: speaker-disjoint
```

The manifests contain **absolute paths**, so they should be generated on the machine that will perform training.

---

# 3. Pairing Guard

The dataset uses a paired design.

Each spoof utterance should correspond to a bonafide utterance with the same underlying text.

This helps prevent the detector from simply learning differences in linguistic content between the classes.

The pairing guard removes spoof clips for which no corresponding bonafide clip exists.

For the current corpus, the expected result is:

```text
Pairing guard: excluded 367 of 56653 spoof clips (0.65%) with no bonafide counterpart.
```

Expected breakdown:

```text
meta-mms        357
elevenlabs       10
```

After the guard:

```text
Bonafide: 56,286
Spoof:    56,286
```

This gives a:

```text
1:1 bonafide-to-spoof ratio
```

### Important

If the number of excluded clips differs from the expected value for the current corpus, investigate the upstream dataset before training.

The pairing guard provides:

```text
--unpaired-report PATH
--max-unpaired-frac F
--allow-unpaired
```

Defaults include a maximum unpaired fraction of:

```text
0.05
```

`--max-unpaired-frac` should be treated as a safety mechanism rather than a tuning parameter.

---

# 4. Shortcut Probe

The shortcut probe measures how much of the classification task can be solved using acoustic/file-level features without actually listening to the speech.

Run:

## Linux

```bash
python training/probe_shortcuts.py \
  --splits-dir manifests/splits_norm \
  --eval-split test \
  --workers 4 \
  --cache outputs/probe_test.parquet
```

## Windows

```powershell
python training\probe_shortcuts.py `
  --splits-dir manifests\splits_norm `
  --eval-split test `
  --workers 4 `
  --cache outputs\probe_test.parquet
```

The important comparison is between:

```text
shortcut probe EER
```

and:

```text
trained detector EER
```

A low shortcut-probe EER indicates that some class information may be available through non-content acoustic characteristics.

The current corpus produced a shortcut probe result that was considered a **serious shortcut risk**, particularly around loudness and silence-related features.

This should be considered when interpreting model results.

---

# 5. DeepFense Preflight

Before training:

```bash
python training/deepfense_patches.py --verify
```

A successful verification should report:

```text
PATCH 1 -- transform pipeline
PASS

PATCH 2 -- collate function
PASS

PATCH 2 -- rebind check
PASS

RESULT: all patches good -- num_workers > 0 is safe.
```

The patches address runtime issues in DeepFense and enable gradient clipping.

Do not skip this step.

---

# 6. Train

Training should be launched through:

```bash
python training/run_training.py \
  --config training/configs/train_main.yaml
```

Do **not** launch the experiment using:

```bash
deepfense train
```

The project's `run_training.py` performs project-specific registration and patching required by the pipeline.

Near the beginning of the training log, verify that you see something similar to:

```text
Applied deepfense patches: ...
Gradient clipping enabled: max_grad_norm=5.0
```

If gradient clipping is missing, stop and investigate before continuing.

---

# Training Configurations

Available configurations:

| Config                | Epochs | Purpose                                                          |
| --------------------- | -----: | ---------------------------------------------------------------- |
| `smoke.yaml`          |      1 | Verify the pipeline with a frozen frontend                       |
| `smoke_unfrozen.yaml` |      1 | Verify joint fine-tuning fits the available GPU memory           |
| `timing_2epoch.yaml`  |      5 | Short timing/training test using the real learning-rate schedule |
| `train_main.yaml`     |     30 | Main training run                                                |

The short timing configuration intentionally uses:

```text
T_max: 30
```

even though it runs fewer epochs.

This keeps the learning-rate schedule comparable to the first part of the full 30-epoch run.

---

# Experiment Matrix

Generate experiment variants using the provided scripts.

```bash
python training/make_loso_splits.py \
  --splits-dir manifests/splits_norm \
  --out manifests/loso
```

Then:

```bash
python training/make_ablation_configs.py --apply
```

Experiments include:

| Run             | Purpose                                   |
| --------------- | ----------------------------------------- |
| `train_main`    | Main experiment                           |
| `noaug`         | Evaluate the effect of codec augmentation |
| `opus`          | Train using Opus augmentation             |
| `amr`           | Train using AMR augmentation              |
| `loso_<system>` | Leave-one-synthesis-system-out evaluation |

Run experiments **one at a time** when using a GPU with limited memory.

Cross-condition evaluation does not require another training run.

A single trained checkpoint can be evaluated against different test conditions such as:

```text
Clean
Opus
AMR
```

---

# What to Monitor

## 1. Validation EER

Monitor validation EER from the first epoch.

Compare the model's EER with the shortcut-probe floor.

EER is the primary metric for this project.

---

## 2. Gradient clipping

Look for:

```text
[grad-clip]
```

Frequent clipping can indicate that the learning rate may be aggressive for the dataset.

---

## 3. Overfitting

A common warning sign is:

```text
training loss ↓
validation loss ↑
```

The training configuration includes early stopping.

---

## 4. Accuracy

Do not use raw `ACC` as the primary performance metric.

The project uses OC-Softmax scoring, where the default threshold does not necessarily provide a meaningful accuracy measurement.

Report:

```text
EER
```

as the primary detection metric.

---

# Long-Running Training

A long training run should not depend on an active SSH connection.

If working on a remote Linux server, use an appropriate persistent session mechanism such as:

```text
tmux
```

or:

```text
screen
```

if available on the server.

For example:

```bash
tmux new -s deepfense
```

Then start training:

```bash
python training/run_training.py \
  --config training/configs/train_main.yaml
```

Detach with:

```text
Ctrl+B
D
```

Reconnect later:

```bash
tmux attach -t deepfense
```

### Important

Do not assume that closing your laptop or disconnecting SSH will preserve a normal foreground process.

Before starting a long training run, verify that your session/job mechanism is persistent.

---

# Storage Requirements

Training checkpoints are large.

Each checkpoint is approximately:

```text
~3.8 GB
```

A 30-epoch run can produce approximately:

```text
40–60 GB
```

depending on how many checkpoints are retained.

Before starting a long run:

```bash
df -h
```

and, if needed:

```bash
du -sh outputs/
```

Keep:

```text
train.log
results.json
```

for experiments that matter.

These files are much smaller than the model checkpoints and provide the record of the experiment.

---

# GPU Requirements

GPU memory is a major constraint for the training configuration.

The tested setup uses approximately:

```text
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16
```

Batch size 8 has been observed to cause:

```text
torch.OutOfMemoryError
```

on the tested 8 GB VRAM environment.

### Do not run other heavy GPU workloads during training.

Games, other training processes, and GPU-heavy applications can consume several GB of VRAM.

Two simultaneous training runs can exceed the available GPU memory.

Run experiments sequentially unless the target machine has enough GPU memory to safely support concurrent jobs.

---

# SVC1 Example Setup

This section documents the setup used on the university SVC1 server.

It is **specific to SVC1** and should not be assumed to apply to every Linux machine.

## SSH

SVC1 uses a custom SSH port:

```text
Host: svc1.dcism.org
Port: 22081
User: <your SVC1 username>
```

Example:

```bash
ssh -p 22081 <your-username>@svc1.dcism.org
```

---

## Storage

SVC1 provides user storage through:

```text
Home directory
~/archive
```

On the tested account:

```text
~
└── archive -> /disk2/storage/users/<username>
```

The home directory is located on the NVMe filesystem, while `~/archive` points to the user's personal HDD storage.

Check:

```bash
df -h ~ ~/archive
```

Do not assume that `/disk2` itself is writable.

Use the user-specific `~/archive` path instead.

---

## Recommended SVC1 layout

A practical arrangement is:

```text
/data/users/<username>/
│
├── cebuano-audio-deepfake-detector/    # code + environment-related files
│
├── miniconda3/                         # Conda installation
│
├── bin/                                # user-local binaries such as FFmpeg
│
└── archive/
    └── normalised/                     # large dataset
        ├── bonafide/
        ├── meta-mms/
        └── elevenlabs/
```

This keeps active code/tools on the faster home/NVMe storage while allowing the large dataset to use the personal archive/HDD area.

---

## Check the GPU

```bash
nvidia-smi
```

Before starting training, make sure another heavy job is not already using the GPU.

Do not kill processes belonging to other users.

---

## Verify the environment

After reconnecting:

```bash
conda activate deepfense
```

Then:

```bash
cd ~/cebuano-audio-deepfake-detector
```

Check:

```bash
python --version
which ffmpeg
nvidia-smi
```

Then:

```bash
python training/codec_opus_amr.py --check-ffmpeg
```

and:

```bash
python training/deepfense_patches.py --verify
```

---

# Troubleshooting

## `ffmpeg: command not found`

If you do not have administrator privileges, install FFmpeg somewhere inside your home directory.

For example:

```text
~/bin/ffmpeg
```

Then:

```bash
export PATH="$HOME/bin:$PATH"
```

To make it persistent:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
which ffmpeg
```

---

## AMR-NB encoder is missing

If the preflight reports:

```text
AMR-NB encoder 'libopencore_amrnb': MISSING
```

the FFmpeg binary does not contain the required encoder.

Check:

```bash
ffmpeg -encoders | grep -i amr
```

and:

```bash
ffmpeg -buildconf | grep -i opencore
```

You need an FFmpeg build containing:

```text
libopencore_amrnb
```

Do not begin the main training run until this is fixed.

---

## `conda-forge` cannot be accessed

Some institutional servers may block or intercept access to Conda repositories.

If Conda package installation fails with network or HTTP errors, use the package/environment sources available to your institution or install required tools into user-local directories where appropriate.

Do not assume that `conda-forge` is accessible on every server.

---

## `CUDA: False`

Check:

```bash
nvidia-smi
```

Then:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the GPU is visible to `nvidia-smi` but not PyTorch, verify that the installed PyTorch build supports the machine's architecture and CUDA environment.

---

## Manifest paths are incorrect

Manifest files contain absolute paths.

If the dataset is moved to another machine, rebuild the manifests there.

Do not simply copy manifests generated on another computer and expect the paths to remain valid.

---

## Linux path expansion issue

This may fail to expand the home directory correctly:

```bash
--spoof-root meta-mms=~/archive/normalised/meta-mms
```

Use:

```bash
--spoof-root "meta-mms=$HOME/archive/normalised/meta-mms"
```

instead.

---

## Training fails with an AMR transform error

An error such as:

```text
AMR-NB augmentation requires the 'libopencore_amrnb' encoder
```

means the FFmpeg binary does not contain the required AMR-NB encoder.

Run:

```bash
python training/codec_opus_amr.py --check-ffmpeg
```

and fix FFmpeg before retrying.

---

## SSH disconnects during training

Do not rely on a normal SSH session for long experiments.

Use:

```text
tmux
```

or:

```text
screen
```

or the scheduler provided by your institution.

---

# Important Methodology Notes

## Paired dataset design

The spoof dataset is designed so that synthetic utterances correspond to bonafide utterances with the same underlying text.

This reduces the possibility that the detector learns content differences instead of synthesis artifacts.

The pairing guard therefore intentionally removes spoof samples without valid bonafide counterparts.

---

## Speaker-disjoint evaluation

Train, validation, and test sets are split by speaker.

This prevents the same speaker from appearing across multiple splits and reduces speaker-identity leakage.

Always confirm:

```text
RESULT: speaker-disjoint
```

in the generated split summary.

---

## Shortcut risk

The shortcut probe exists to determine whether classification can be performed using simple acoustic or file-level properties.

A strong shortcut-probe result does not automatically invalidate the experiment, but it is important when interpreting the detector's performance.

The current corpus showed meaningful separability using non-content features, particularly features related to loudness and silence.

Normalization and identical preprocessing across classes are therefore important parts of the pipeline.

See:

```text
METHODOLOGY_AUDIT.md
```

for the detailed methodology discussion.

---

# Recommended Pre-Training Checklist

Before starting a long training run, verify all of the following:

```text
[ ] Correct repository checked out
[ ] Correct dataset uploaded
[ ] Bonafide dataset present
[ ] Meta MMS dataset present
[ ] ElevenLabs dataset present

[ ] Manifests generated on the training machine
[ ] Train/val/test split is speaker-disjoint
[ ] Pairing guard completed
[ ] Expected pairing count investigated

[ ] Shortcut probe completed
[ ] FFmpeg installed
[ ] libopus available
[ ] libopencore_amrnb available

[ ] Correct Conda/venv activated
[ ] PyTorch sees CUDA
[ ] Correct GPU selected
[ ] DeepFense patches verified

[ ] Enough disk space available
[ ] No competing heavy GPU jobs
[ ] Persistent training session configured
```

Then:

```bash
python training/run_training.py \
  --config training/configs/train_main.yaml
```

---

# Quick Start — Linux

For a Linux machine where the dataset and repository are already available:

```bash
# Activate environment
conda activate deepfense

# Enter repository
cd ~/cebuano-audio-deepfake-detector

# Check GPU
nvidia-smi

# Check FFmpeg
which ffmpeg
python training/codec_opus_amr.py --check-ffmpeg

# Verify dataset
du -sh ~/archive/normalised

# Build manifests
python training/build_manifests.py \
  --bonafide-root "$HOME/archive/normalised/bonafide" \
  --spoof-root "meta-mms=$HOME/archive/normalised/meta-mms" \
  --spoof-root "elevenlabs=$HOME/archive/normalised/elevenlabs" \
  --speaker-manifest manifests/manifest_bonafide.csv \
  --out manifests/splits_norm \
  --unpaired-report manifests/splits_norm/unpaired_excluded.csv

# Shortcut probe
python training/probe_shortcuts.py \
  --splits-dir manifests/splits_norm \
  --eval-split test \
  --workers 4 \
  --cache outputs/probe_test.parquet

# DeepFense preflight
python training/deepfense_patches.py --verify

# Train
python training/run_training.py \
  --config training/configs/train_main.yaml
```

For long-running training, start the final command inside `tmux`, `screen`, or the appropriate cluster job system.

---

# Quick Start — Windows || YES

```powershell
py -3.11 -m venv venv
venv\Scripts\activate

pip install soundfile librosa pandas numpy
pip install wandb scikit-learn
pip install transformers accelerate scipy
pip install elevenlabs python-dotenv
```

Install a suitable FFmpeg build containing:

```text
libopus
libopencore_amrnb
libmp3lame
```

Then run the pipeline:

```powershell
python training\build_manifests.py ...
python training\probe_shortcuts.py ...
python training\deepfense_patches.py --verify
python training\run_training.py --config training\configs\train_main.yaml
```

---

# Final Notes

The most important rule is:

> **Follow the pipeline in order, verify every preflight, and do not start a long GPU run until the environment, codecs, manifests, storage, and GPU availability have all been checked.**

The project-specific methodology and experimental decisions are documented separately in:

```text
METHODOLOGY_AUDIT.md
```
