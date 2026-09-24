# A Guide For Cliff :>

Install Python and Jupyter extensions in VSCode
Your Python version should be 3.11

Do `py -3.11 -m venv venv` in terminal to create venv

Do `venv\Scripts\activate` _ALWAYS_ when working in Python

Install PyTorch

- run `nvidia-smi` in your terminal
- look for CUDA version
  ![alt text](image-1.png)
- Select build (focus on Compute Platform and find the version that's <= your CUDA version)
  ![alt text](image-2.png)
- install using the command (see _Run This Command_)

Then

```
pip install soundfile librosa pandas numpy` // for processing
pip install wandb scikit-learn` // for general ML
pip install transformers accelerate scipy` // for Meta MMS
pip install elevenlabs python-dotenv
```

`winget install ffmpeg` on your terminal

Install Miniconda https://www.anaconda.com/download/success
![alt text](image.png)

Then do

```
conda create -n deepfense python=3.10
conda activate deepfense
pip install deepfense
```


---

# Training

Everything below assumes the venv is active (`venv\Scripts\activate`)
and you are in the repo root.

## Before anything

```powershell
$env:PYTHONUTF8=1
```

deepfense logs emoji. Without this, a cp1252 console throws
`UnicodeEncodeError` on every metrics line. Training is unaffected but
the output is unreadable. `setx PYTHONUTF8 1` once to make it stick.

ffmpeg needs **libmp3lame**, **libopus** and **libopencore_amrnb**
compiled in — not just installed as libraries. Check:

```powershell
python training\codec_opus_amr.py --check-ffmpeg
python training\normalise_corpus.py --check-ffmpeg
```

If AMR is missing: `winget install Gyan.FFmpeg.Full`.

## The pipeline, in order

Each step depends on the one before it. You only re-run from the point
something changed.

### 1. Normalise the corpus

Puts every clip through one identical chain so the class label stops
being readable from production artefacts. See `METHODOLOGY_AUDIT.md`
for why this exists.

```powershell
python training\normalise_corpus.py --bonafide-root data\processed\bonafide --spoof-root meta-mms=data\processed\meta-mms --spoof-root elevenlabs=data\processed\elevenlabs --out-root data\normalised --apply --workers 6
```

Dry run first (drop `--apply`) — it processes a sample in memory and
prints before/after levels per source. ~1.5 h for 113k files. Not
resumable; if it dies you start over.

### 2. Build the manifests

Speaker-disjoint, gender-stratified 70/15/15. Writes **absolute
paths**, so it must run on the machine that will train.

```powershell
python training\build_manifests.py --bonafide-root data\normalised\bonafide --spoof-root meta-mms=data\normalised\meta-mms --spoof-root elevenlabs=data\normalised\elevenlabs --speaker-manifest manifests\manifest_bonafide.csv --out manifests\splits_norm
```

Read `manifests\splits_norm\split_summary.txt` and confirm the leakage
check says `speaker-disjoint`.

### 3. Measure the shortcut floor

How much of the task is solvable *without listening to the speech*.
This is the control the detector's EER gets compared against.

```powershell
python training\probe_shortcuts.py --splits-dir manifests\splits_norm --eval-split test --workers 4 --cache outputs\probe_test.parquet
```

`--splits-dir` is required on purpose — it used to default to
`manifests\splits`, which silently measured the old un-normalised
corpus and returned a plausible wrong answer.

Use `--workers 1` if training is running: each worker loads its own
numpy/scipy and Windows will exhaust the pagefile, reporting it as a
confusing DLL import error.

### 4. Preflight

```powershell
python training\deepfense_patches.py --verify
```

Three runtime patches for deepfense bugs that only bite on Windows —
two unpicklable closures that break DataLoader workers, plus gradient
clipping, which deepfense does not have at all. Details in that file's
docstring.

### 5. Train

```powershell
python training\run_training.py --config training\configs\train_main.yaml
```

**Use `run_training.py`, never `deepfense train` directly.** The bare
CLI does not import `codec_opus_amr`, so the custom transforms are
unregistered and it fails with `KeyError: 'opus_codec'`. It also does
not apply the patches.

Confirm near the start:

```
Applied deepfense patches: ...
Gradient clipping enabled: max_grad_norm=5.0
```

If the clipping line is missing, the patch did not take.

## Configs

| config | epochs | purpose |
|---|---|---|
| `smoke.yaml` | 1 | pipeline works at all, frozen frontend |
| `smoke_unfrozen.yaml` | 1 | does joint fine-tuning fit in 8 GB |
| `timing_2epoch.yaml` | 5 | short test at the real learning rate |
| `train_main.yaml` | 30 | the real run |

`timing_2epoch.yaml` sets `T_max: 30` while running 5 epochs, on
purpose. Cosine decay is defined over `T_max`, so matching it to a
short epoch count collapses the LR to 25% by epoch 3 and the test stops
resembling the real run. With `T_max: 30` its five epochs see exactly
the learning rates the real run's first five will see.

## The experiment matrix

Generate the variants rather than copying the file by hand — the
generator diffs each output against the base and refuses to write one
that differs in anything beyond its declared keys.

```powershell
python training\make_loso_splits.py --splits-dir manifests\splits_norm --out manifests\loso
python training\make_ablation_configs.py --apply
```

| run | answers |
|---|---|
| `train_main` | the headline number |
| `noaug` | what codec augmentation buys |
| `opus` / `amr` | does training on one codec transfer to the other |
| `loso_<system>` | does it generalise to a synthesis system never seen |

**Run them one at a time.** Two concurrent runs will OOM an 8 GB card.

Cross-*condition* evaluation (clean vs Opus vs AMR at test time) does
**not** need its own training run — score one checkpoint against
differently degraded test sets.

## What to watch

- **Epoch 1 validation EER against the probe floor.** Below it by a
  wide margin means the detector is using something the handcrafted
  features cannot. That comparison is the whole point.
- **`[grad-clip]` lines.** Frequent means the LR is too high for this
  data. Absent means a divergence was not a gradient spike.
- **Train loss down while val loss goes up** — overfitting, and early
  stopping (patience 7) will handle it.
- **`ACC` is meaningless here.** It sits at the bonafide class prior
  because OC-Softmax scores are not calibrated around the default
  threshold. Report EER, which is threshold-free.

## Disk

**Each checkpoint is ~3.8 GB** — the file carries Adam optimiser state
alongside the model — and one is written on **every** validation
improvement. A 30-epoch run can produce 40–60 GB. `outputs/` is
gitignored for this reason.

Before a long run, check free space, and copy `train.log` and
`results.json` out of any run you care about before clearing it. They
are a few KB and they are the only record of the trajectory.

## Hardware

8 GB VRAM is the binding constraint. Batch 4 with accumulation 4
(effective 16) is the proven envelope for unfrozen XLS-R; batch 8
throws `torch.OutOfMemoryError`.

Do not run anything else heavy on the GPU during a run. A game claims
2–4 GB of VRAM at launch regardless of graphics preset, and a driver
reset will kill a run that is a day deep.
