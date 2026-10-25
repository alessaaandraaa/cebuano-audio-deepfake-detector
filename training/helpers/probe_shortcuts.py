r"""
probe_shortcuts.py

Measures how much of the detector's performance could come from
production artefacts rather than from speech.

THE PROBLEM

Every bonafide clip came off a recording chain: a person, a room, a
microphone, then whatever the corpus distributor did to it. Every spoof
clip came out of a program writing an audio file. Those two production
lines leave fingerprints that have nothing to do with language --
loudness normalisation, true-zero digital silence, the absence of a
noise floor, the rolloff left behind by downsampling from 44.1 kHz.

All of it correlates perfectly with the label, and all of it is far
easier to learn than pronunciation. A model that reaches a low EER by
reading those cues has learned which pipeline produced the file, not
whether a human spoke. The EER alone cannot tell the two apart.

THE TEST

Fit a deliberately weak classifier -- logistic regression on a handful
of handcrafted numbers -- on features that contain no linguistic
information whatsoever. No words, no phonemes, no speaker identity, no
prosody. Just loudness, silence, noise floor, spectral shape.

This model is structurally incapable of detecting a deepfake. So
whatever EER it reaches is a floor on how much of the task is solvable
without listening to the speech at all.

    probe EER near 0.50   No usable shortcut. The detector's number is
                          doing real work, and you can now say so with
                          evidence instead of assertion.

    probe EER 0.25-0.40   Mild leakage. Report it, and note the
                          detector beats it by a wide margin.

    probe EER 0.10-0.25   Serious. A meaningful share of the
                          detector's EER is channel, not speech.

    probe EER below 0.10  The shortcut nearly solves the task by
                          itself. The headline EER does not mean what
                          the paper will claim it means.

Also reports, separately, the FILE PROVENANCE metadata (native sample
rate, bit depth, channels) broken down by class. If bonafide and spoof
differ systematically there, that is the smoking gun and it is worth
fixing before anything else.

WHAT THIS IS NOT

It is not an ablation and not a baseline detector. It is a validity
check: it bounds what the headline number can honestly be attributed
to. If the probe comes back clean, this is a paragraph in Section 4.10
that pre-empts the first question a reviewer will ask.

COST

CPU only, no GPU, no model loading, so it never contends for the card.
It does contend for RAM: every worker process loads its own numpy and
scipy. If training is running -- four dataloader workers, pinned host
buffers, a 300M-parameter model -- use --workers 1, or Windows will
run out of pagefile and report it as a confusing DLL import failure.
The script catches that and falls back to one process by itself, but
starting with 1 saves the restart.

Roughly 10-30 minutes on the default 8000-clip subsample. Re-runs are
near-instant because features are cached.

Usage:

    # while training is running -- one process, low memory
    python training\probe_shortcuts.py --splits-dir manifests\splits_norm --workers 1

    # machine otherwise idle
    python training\probe_shortcuts.py --splits-dir manifests\splits_norm --workers 4

    # everything, no subsample (idle machine only)
    python training\probe_shortcuts.py --splits-dir manifests\splits_norm --max-per-split 0 --workers 4

    # re-analyse instantly from the cache, no audio touched
    python training\probe_shortcuts.py --from-cache --cache outputs\probe_test.parquet

    # THE NUMBER YOU REPORT: fit on train, score on TEST, compared
    # against the detector's own test EER on the same split
    python training\probe_shortcuts.py --splits-dir manifests\splits_norm --eval-split test --workers 4 --cache outputs\probe_test.parquet --model-eer <detector test EER>
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_SR = 16000

# Frame geometry for the noise-floor / dynamic-range features.
FRAME_LEN = 400  # 25 ms @ 16 kHz
FRAME_HOP = 160  # 10 ms @ 16 kHz

# Features that describe the whole file rather than its content. The
# model never sees these -- it gets a fixed ~4 s crop -- so they are
# reported separately and excluded from the headline probe.
WHOLE_FILE_FEATURES = ["duration_s", "lead_silence_s", "trail_silence_s"]

FEATURE_NOTES = {
    "duration_s": "clip length (model sees a fixed 4 s crop)",
    "rms_db": "overall loudness",
    "peak_db": "peak level -- near 0 dB suggests normalisation",
    "crest_db": "peak minus RMS",
    "dc_offset": "DC bias, a recording-chain artefact",
    "zcr": "zero-crossing rate",
    "frac_exact_zero": "TRUE digital silence -- mics never produce it",
    "lead_silence_s": "leading zero-run",
    "trail_silence_s": "trailing zero-run",
    "noise_floor_db": "quietest frames -- room tone vs synthetic",
    "dyn_range_db": "loud frames minus quiet frames",
    "hf_ratio_7k": "energy above 7 kHz",
    "band_7k_8k_ratio": "energy in 7-8 kHz -- downsampling rolloff",
    "rolloff_95_hz": "frequency holding 95% of the energy",
    "spectral_flatness": "noise-like vs tonal",
    "clip_frac": "samples at full scale",
}


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------


def compute_eer(labels, scores):
    """
    Equal error rate, with bonafide (1) as the target class.

    Returns (eer, auc). Implemented here rather than imported so the
    probe stays independent of deepfense's evaluation code -- an
    independent check should not share machinery with the thing it is
    checking.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg
    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    auc = (
        float(np.trapezoid(tpr, fpr))
        if hasattr(np, "trapezoid")
        else (float(np.trapz(tpr, fpr)))
    )

    return eer, auc


def directional_eer(labels, values):
    """
    EER of a single raw feature used directly as a score.

    A feature can separate the classes in either direction, so both are
    tried and the better one is kept. Returns (eer, auc, sign) where
    sign is +1 if higher values indicate bonafide.
    """
    finite = np.isfinite(values)

    if finite.sum() < 10:
        return float("nan"), float("nan"), 0

    lab = np.asarray(labels)[finite]
    val = np.asarray(values, dtype=float)[finite]

    eer_pos, auc_pos = compute_eer(lab, val)
    eer_neg, auc_neg = compute_eer(lab, -val)

    if eer_pos <= eer_neg:
        return eer_pos, auc_pos, +1

    return eer_neg, auc_neg, -1


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------


def _resample(x, sr_in, sr_out=TARGET_SR):
    """Polyphase resample. Matches what the loader does closely enough."""
    if sr_in == sr_out:
        return x

    from scipy.signal import resample_poly

    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)

    return resample_poly(x, ratio.numerator, ratio.denominator)


def _frame_rms(x):
    """RMS of each 25 ms frame, hop 10 ms."""
    if len(x) < FRAME_LEN:
        return np.array([np.sqrt(np.mean(x**2) + 1e-20)])

    n = 1 + (len(x) - FRAME_LEN) // FRAME_HOP
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_HOP * np.arange(n)[:, None]

    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-20)


def _db(v, floor=1e-10):
    return float(20.0 * np.log10(max(float(v), floor)))


def extract_features(path):
    """
    All features for one clip.

    Signal features are computed on the 16 kHz mono version, because
    that is what the model actually sees -- but the traces of an
    earlier downsample (a rolloff below Nyquist) survive resampling,
    which is exactly what makes them a usable shortcut.

    Provenance fields come from the file header and are diagnostic
    only; they never enter the classifier.
    """
    import soundfile as sf

    out = {"path": path}

    try:
        info = sf.info(path)
        out["native_sr"] = int(info.samplerate)
        out["subtype"] = str(info.subtype)
        out["channels"] = int(info.channels)
        out["container"] = str(info.format)

        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    if x.ndim > 1:
        x = x.mean(axis=1)

    out["duration_s"] = float(len(x) / sr) if sr else 0.0

    x = _resample(np.asarray(x, dtype=np.float64), sr)

    if len(x) < FRAME_LEN:
        out["error"] = f"too short after resample: {len(x)} samples"
        return out

    absx = np.abs(x)

    # --- level -----------------------------------------------------
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(absx.max())

    out["rms_db"] = _db(rms)
    out["peak_db"] = _db(peak)
    out["crest_db"] = out["peak_db"] - out["rms_db"]
    out["dc_offset"] = float(np.mean(x))
    out["clip_frac"] = float(np.mean(absx >= 0.999))

    # --- silence ---------------------------------------------------
    # Exact zeros are the giveaway: a microphone chain essentially
    # never produces a run of bit-exact zeros, a synthesiser does.
    is_zero = absx == 0.0
    out["frac_exact_zero"] = float(np.mean(is_zero))

    nz = np.flatnonzero(~is_zero)

    if nz.size:
        out["lead_silence_s"] = float(nz[0] / TARGET_SR)
        out["trail_silence_s"] = float((len(x) - 1 - nz[-1]) / TARGET_SR)
    else:
        out["lead_silence_s"] = out["duration_s"]
        out["trail_silence_s"] = out["duration_s"]

    out["zcr"] = float(np.mean(np.diff(np.signbit(x)) != 0))

    # --- noise floor / dynamics ------------------------------------
    frames = _frame_rms(x)
    p10 = _db(np.percentile(frames, 10))
    p95 = _db(np.percentile(frames, 95))

    out["noise_floor_db"] = p10
    out["dyn_range_db"] = p95 - p10

    # --- spectrum --------------------------------------------------
    # Welch rather than a single FFT: averaged periodograms are far
    # less noisy for the band-ratio features, which are the ones most
    # likely to expose sample-rate provenance.
    from scipy.signal import welch

    nperseg = min(1024, len(x))
    freqs, psd = welch(x, fs=TARGET_SR, nperseg=nperseg)

    total = float(psd.sum()) + 1e-20

    out["hf_ratio_7k"] = float(psd[freqs >= 7000].sum() / total)
    out["band_7k_8k_ratio"] = float(
        psd[(freqs >= 7000) & (freqs < 8000)].sum() / total
    )

    cumulative = np.cumsum(psd) / total
    out["rolloff_95_hz"] = float(freqs[np.searchsorted(cumulative, 0.95)])

    positive = psd[psd > 0]

    if positive.size:
        geo = float(np.exp(np.mean(np.log(positive))))
        out["spectral_flatness"] = geo / (float(positive.mean()) + 1e-20)
    else:
        out["spectral_flatness"] = 0.0

    return out


# ---------------------------------------------------------------------------
# MANIFEST HANDLING
# ---------------------------------------------------------------------------


def load_split(splits_dir, name):
    """Read one split parquet and normalise the label column to 0/1."""
    path = Path(splits_dir) / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the manifests first with "
            "build_manifests.py."
        )

    df = pd.read_parquet(path)

    if "label" not in df.columns:
        raise ValueError(f"{path} has no 'label' column: {list(df.columns)}")

    labels = df["label"]

    # Parquet round-trips strings as object, pandas "string", or
    # category depending on the writer, so test for numeric rather
    # than for any one string dtype.
    if pd.api.types.is_numeric_dtype(labels):
        df["y"] = labels.astype(int)
    else:
        mapping = {"bonafide": 1, "spoof": 0}
        values = pd.Series(labels).astype(str).str.strip().str.lower()
        unknown = set(values.unique()) - set(mapping)

        if unknown:
            raise ValueError(f"unexpected label values: {sorted(unknown)}")

        df["y"] = values.map(mapping).astype(int)

    if "spoof_method" not in df.columns:
        df["spoof_method"] = None

    df["split"] = name

    return df


def subsample(df, n, seed):
    """Class-balanced subsample. n <= 0 keeps everything."""
    if n <= 0 or len(df) <= n:
        return df

    per_class = max(1, n // 2)
    parts = []

    for y, group in df.groupby("y"):
        take = min(per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))

    return pd.concat(parts).sample(frac=1.0, random_state=seed)


# ---------------------------------------------------------------------------
# REPORT SECTIONS
# ---------------------------------------------------------------------------


def rule(char="-", width=72):
    print(char * width)


def report_provenance(df):
    """
    Header metadata by class.

    This runs first and deliberately stands apart from the classifier:
    if the two classes differ here, no amount of modelling
    sophistication makes the result trustworthy, and the fix is a
    preprocessing change rather than a training one.
    """
    rule("=")
    print("FILE PROVENANCE (header metadata, by class)")
    rule("=")
    print()

    flagged = []

    for col in ("native_sr", "subtype", "channels", "container"):
        if col not in df.columns:
            continue

        table = pd.crosstab(df[col], df["y"])
        table.columns = [
            {0: "spoof", 1: "bonafide"}.get(c, str(c)) for c in table.columns
        ]

        print(f"{col}:")
        print(table.to_string())

        # A value that occurs almost exclusively in one class is a
        # label the model can read straight off the file.
        share = table.div(table.sum(axis=1), axis=0)
        lopsided = share[(share > 0.95).any(axis=1)]

        if len(lopsided) and len(table) > 1:
            flagged.append(col)
            print(f"  >> WARNING: some {col} values are >95% one class")

        print()

    if flagged:
        print("PROVENANCE VERDICT: LEAKAGE PRESENT in " + ", ".join(flagged))
        print()
        print(
            "  The class is partly readable from file metadata alone.\n"
            "  Resampling at load time does not erase this -- a\n"
            "  downsampled 44.1 kHz file keeps a rolloff that a native\n"
            "  16 kHz file does not have. Fix by normalising every\n"
            "  file through the same chain before training."
        )
    else:
        print("PROVENANCE VERDICT: clean -- no class-specific metadata.")

    print()


def report_univariate(train, feat_cols):
    """Rank features by how well each one separates the classes alone."""
    rule("=")
    print("SINGLE-FEATURE SEPARABILITY (train split)")
    rule("=")
    print()
    print("  Each feature used on its own as the score. An EER near")
    print("  0.50 means that feature tells you nothing.")
    print()
    print(f"  {'feature':<20} {'EER':>7} {'AUC':>7}  note")
    rule()

    rows = []

    for col in feat_cols:
        eer, auc, _ = directional_eer(train["y"].values, train[col].values)
        rows.append((eer, auc, col))

    for eer, auc, col in sorted(rows):
        mark = "  <<" if eer < 0.30 else ""
        note = FEATURE_NOTES.get(col, "")
        print(f"  {col:<20} {eer:>7.4f} {auc:>7.4f}  {note}{mark}")

    print()
    print("  << marks a feature that alone reaches EER < 0.30.")
    print()

    return rows


def fit_probe(train, val, cols, label):
    """Logistic regression on `cols`, scored on val."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_tr = train[cols].to_numpy(dtype=float)
    x_va = val[cols].to_numpy(dtype=float)

    # Median-impute from train only; a non-finite value is itself
    # informative but should not silently become a feature.
    medians = np.nanmedian(np.where(np.isfinite(x_tr), x_tr, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    x_tr = np.where(np.isfinite(x_tr), x_tr, medians)
    x_va = np.where(np.isfinite(x_va), x_va, medians)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    model.fit(x_tr, train["y"].to_numpy())

    scores = model.predict_proba(x_va)[:, 1]
    eer, auc = compute_eer(val["y"].to_numpy(), scores)

    print(f"  {label:<34} EER {eer:.4f}   AUC {auc:.4f}")

    return eer, scores, model


def report_by_source(val, scores):
    """Is one spoof system easier to shortcut than the other?"""
    if val["spoof_method"].isna().all():
        return

    rule()
    print("  By spoof system (bonafide vs each, same probe scores):")

    bona = val["y"] == 1

    for method in sorted(val.loc[~bona, "spoof_method"].dropna().unique()):
        mask = bona | (val["spoof_method"] == method)
        eer, _ = compute_eer(
            val.loc[mask, "y"].to_numpy(), scores[mask.values]
        )
        print(f"    bonafide vs {method:<20} EER {eer:.4f}")

    print()


def report_verdict(eer, model_eer):
    rule("=")
    print("VERDICT")
    rule("=")
    print()

    if not np.isfinite(eer):
        print("  Probe EER could not be computed.")
        return 1

    if eer >= 0.40:
        band = "CLEAN"
        text = (
            "No usable shortcut. Features carrying no linguistic\n"
            "  information barely beat chance, so the detector's number\n"
            "  is attributable to the speech itself. Report this."
        )
    elif eer >= 0.25:
        band = "MILD"
        text = (
            "Some production signal is present but weak. Report the\n"
            "  probe alongside the detector and note the margin."
        )
    elif eer >= 0.10:
        band = "SERIOUS"
        text = (
            "A meaningful share of the task is solvable without\n"
            "  listening to the speech. Normalise loudness, trim\n"
            "  silence consistently across both classes, and confirm\n"
            "  every file goes through an identical resampling chain."
        )
    else:
        band = "CRITICAL"
        text = (
            "The shortcut nearly solves the task by itself. The\n"
            "  headline EER does not measure deepfake detection in any\n"
            "  defensible sense. Fix preprocessing before running the\n"
            "  experiment matrix."
        )

    print(f"  Probe EER: {eer:.4f}   ->  {band}")
    print()
    print(f"  {text}")
    print()

    if model_eer is not None:
        gap = eer - model_eer
        print(f"  Detector EER: {model_eer:.4f}")
        print(f"  Probe minus detector: {gap:+.4f}")
        print()

        if model_eer >= eer:
            print(
                "  The detector is NO BETTER than the shortcut. Whatever\n"
                "  it learned, it is not beating a model that cannot\n"
                "  hear words. This needs explaining before publication."
            )
        elif gap < 0.05:
            print(
                "  The detector barely beats the shortcut. Most of its\n"
                "  apparent skill may be production artefacts."
            )
        else:
            print(
                "  The detector clearly beats the shortcut, which is the\n"
                "  result you want: it is using something the handcrafted\n"
                "  features cannot capture."
            )

        print()

    return 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def _extract_batch(paths):
    """
    Feature-extract a chunk of paths inside one worker.

    Dispatching per chunk rather than per clip keeps the pending-future
    table small and amortises the cost of handing work to the pool.
    """
    return [extract_features(p) for p in paths]


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _progress(done, total):
    if done % 500 < 1 or done >= total:
        print(f"  {done}/{total}  ({100.0 * done / total:.1f}%)")


def _gather_serial(paths):
    """
    Single-process extraction.

    Slower, but it imports numpy/scipy exactly once and allocates one
    process's worth of memory. This is the path that survives on a
    machine already running a training job.
    """
    records = []

    for i, path in enumerate(paths, 1):
        records.append(extract_features(path))
        _progress(i, len(paths))

    return records


def _gather_parallel(paths, workers):
    """Multi-process extraction. Raises if the pool cannot be sustained."""
    records = []
    batches = list(_chunk(paths, 64))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_batch, b) for b in batches]

        for future in as_completed(futures):
            records.extend(future.result())
            _progress(len(records), len(paths))

    return records


def gather_features(df, workers, cache_path):
    """
    Extract features for every row.

    Each worker process imports its own copy of numpy and scipy, which
    is a few hundred MB apiece. On a machine that is simultaneously
    running training -- four dataloader workers, pinned host buffers,
    a 300M-parameter model -- that is often what pushes Windows past
    its pagefile limit, and it surfaces as

        ImportError: DLL load failed while importing _distance_pybind:
        The paging file is too small for this operation to complete.

    which is a memory error wearing an import error's clothes. When
    that happens there is no point failing: fall back to a single
    process and finish the job slower.
    """
    paths = df["path"].tolist()

    print(
        f"Extracting features from {len(paths)} clips "
        f"using {workers} worker(s)..."
    )

    if workers <= 1:
        records = _gather_serial(paths)
    else:
        try:
            records = _gather_parallel(paths, workers)
        except (ImportError, OSError, MemoryError) as e:
            print()
            print("-" * 72)
            print(f"Worker pool failed: {type(e).__name__}: {e}")
            print()
            print(
                "This is almost always memory pressure, not a code\n"
                "fault -- each worker loads its own numpy/scipy. Falling\n"
                "back to a single process and starting the extraction\n"
                "over. It will be slower but it will finish.\n"
                "\n"
                "To avoid the restart next time, pass --workers 1, or\n"
                "run this when training is not occupying the machine."
            )
            print("-" * 72)
            print()
            records = _gather_serial(paths)

    feats = pd.DataFrame.from_records(records)
    merged = df.merge(feats, on="path", how="left")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        print(f"Cached features -> {cache_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bound how much of the detector's EER is explainable by "
            "production artefacts rather than speech."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--splits-dir",
        required=True,
        help=(
            "directory holding train/val/test.parquet. REQUIRED, with "
            "no default on purpose: this used to default to "
            "manifests/splits, and once the normalised corpus moved to "
            "manifests/splits_norm that default silently re-measured "
            "the OLD leaky corpus and returned a plausible-looking "
            "wrong answer. An explicit path cannot do that."
        ),
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=8000,
        help=(
            "class-balanced subsample per split; 0 uses everything "
            "(default: 8000, enough for a stable EER)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "feature-extraction processes (default: 2). Each one loads "
            "its own numpy/scipy, so raise this only when the machine "
            "is otherwise idle; use 1 while training is running."
        ),
    )
    parser.add_argument(
        "--cache",
        default="outputs/shortcut_probe_features.parquet",
        help="where to write/read extracted features",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="skip audio entirely and re-analyse the cached features",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help=(
            "which split to score the probe on (default: val). Use "
            "test for the number you report: the probe and the "
            "detector must be measured on the same data, or the "
            "comparison between them means nothing."
        ),
    )
    parser.add_argument(
        "--model-eer",
        type=float,
        default=None,
        help=(
            "the detector's val EER, to print the comparison directly "
            "(e.g. --model-eer 0.0435)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="subsample seed (default: 42, matching the split seed)",
    )
    parser.add_argument(
        "--dump-csv",
        default=None,
        help="also write the feature table to this CSV",
    )

    args = parser.parse_args()

    # --- load ----------------------------------------------------------
    if args.from_cache:
        if not Path(args.cache).exists():
            print(f"ERROR: no cache at {args.cache}", file=sys.stderr)
            return 1

        data = pd.read_parquet(args.cache)
        print(f"Loaded {len(data)} cached rows from {args.cache}")
    else:
        try:
            train_raw = load_split(args.splits_dir, "train")
            val_raw = load_split(args.splits_dir, args.eval_split)

            # Relabel so the rest of the script, and the cache, do not
            # care which split was scored.
            val_raw["split"] = "eval"
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        train_raw = subsample(train_raw, args.max_per_split, args.seed)
        val_raw = subsample(val_raw, args.max_per_split, args.seed)

        combined = pd.concat([train_raw, val_raw], ignore_index=True)
        data = gather_features(combined, args.workers, args.cache)

    if args.dump_csv:
        Path(args.dump_csv).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.dump_csv, index=False)
        print(f"Wrote {args.dump_csv}")

    # --- clean ---------------------------------------------------------
    if "error" in data.columns:
        failed = data["error"].notna().sum()

        if failed:
            print(f"WARNING: {failed} clips failed to read; dropping them.")
            for msg in data["error"].dropna().unique()[:5]:
                print(f"    {msg}")

        data = data[data["error"].isna()]

    feat_cols = [c for c in FEATURE_NOTES if c in data.columns]
    core_cols = [c for c in feat_cols if c not in WHOLE_FILE_FEATURES]

    if not core_cols:
        print("ERROR: no usable features extracted.", file=sys.stderr)
        return 1

    train = data[data["split"] == "train"]
    val = data[data["split"] == "eval"]

    if val.empty:
        print(
            "ERROR: no evaluation rows. A cache built by an older "
            "version labels them 'val'; delete the cache and re-run.",
            file=sys.stderr,
        )
        return 1

    print()
    print(
        f"train: {len(train)} clips   "
        f"{getattr(args, 'eval_split', 'eval')}: {len(val)} clips"
    )
    print(
        f"features: {len(core_cols)} core, "
        f"{len(feat_cols) - len(core_cols)} whole-file"
    )
    print()

    # --- report --------------------------------------------------------
    report_provenance(data)
    report_univariate(train, feat_cols)

    rule("=")
    print(
        "SHORTCUT PROBE (fit on train, scored on"
        f" {getattr(args, 'eval_split', 'eval')})"
    )
    rule("=")
    print()

    eer, scores, _ = fit_probe(
        train, val, core_cols, "core features (model-visible)"
    )

    if len(core_cols) < len(feat_cols):
        fit_probe(
            train,
            val,
            feat_cols,
            "+ whole-file features (upper bound)",
        )
        print()
        print("  Whole-file features are shown for context only -- the")
        print("  detector sees a fixed 4 s crop and cannot use them.")

    print()
    report_by_source(val, scores)

    return report_verdict(eer, args.model_eer)


if __name__ == "__main__":
    sys.exit(main())
