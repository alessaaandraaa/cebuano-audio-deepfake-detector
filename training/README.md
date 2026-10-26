# TRAINING

This folder contains all scripts and configs for training and evaluating the detection model.

The model is XLS-R 300M + AASIST with OC-Softmax, trained through DeepFense.

## FOLDERS

**`/configs`** - YAML configs for the Meta MMS corpus runs

**`/configs_vc`** - YAML configs for the FreeVC corpus runs; these are the ones behind the reported results

**`/helpers`** - helper scripts for pre-training checks and corpus accounting

## IMPORTANT SCRIPTS

**`build_manifests.py`**

- builds the train/val/test Parquet manifests using a 70/15/15 speaker-disjoint split, stratified by gender
- writes to `manifests/splits*`

**`run_training.py`**

- launches DeepFense training
- always launch through this and never `deepfense train` directly, since this wrapper is what registers `opus_codec` and `amrnb_codec`

**`codec_opus_amr.py`**

- the Opus and AMR-NB codec augmentation, applied on the fly through ffmpeg so nothing extra is written to disk
- run it with `--self-test` first, since it needs an ffmpeg build that actually has libopus and AMR-NB

## SPLIT AND CONFIG GENERATION

**`make_loso_splits.py`**

- derives the leave-one-system-out manifests, to test whether the detector generalises to a synthesis system it has never seen
- writes to `manifests/loso` and `manifests/loso_vc`

**`make_per_source_eval.py`**

- splits the pooled test set per synthesis system, so that one system cannot quietly carry the result

**`make_ablation_configs.py`**

- generates the ablation configs from the main config, changing exactly one thing each
- generated rather than hand-copied so the arms are guaranteed to differ only in the variable under test

## SCORING SCRIPTS

**`score_predictions.py`**

- computes accuracy, precision, recall and F1 from saved per-sample scores, at a threshold calibrated on a held-out split

**`per_source_eer.py`**

- per-system EER from a saved prediction file, without re-running the model

**`speaker_bootstrap.py`**

- speaker-level bootstrap confidence intervals for EER, and paired differences between two runs
- resamples speakers rather than clips, since clips from the same speaker score alike and clip-level intervals come out too narrow

## RECOMMENDED PIPELINE

1. Run `build_manifests.py` to build the splits
2. Run `helpers/preflight.py` to check the splits before committing GPU time
3. Run `codec_opus_amr.py --self-test` to confirm ffmpeg has both codecs
4. Run `run_training.py` with a config from `/configs_vc`
5. Run `make_per_source_eval.py`, then `score_predictions.py` and `per_source_eer.py` on the saved predictions
6. Run `speaker_bootstrap.py` for the confidence intervals

Use `make_loso_splits.py` for the generalisation experiments, and `make_ablation_configs.py` for the ablation arms.
