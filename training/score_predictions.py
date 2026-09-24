r"""
score_predictions.py

Computes accuracy, precision, recall and F1 from saved per-sample
scores, at a threshold calibrated on a held-out split.

WHY THIS EXISTS

deepfense registers exactly three metrics -- EER, ACC and F1_SCORE
(evaluations/metrics.py). Two of them are unusable as shipped:

  * ACC and F1_SCORE both hardcode `predictions = (scores > 0)`.
    OC-Softmax scores are not calibrated around zero, so every sample
    lands on one side of it and ACC collapses to the class prior. That
    is why every run so far reported ACC 0.4894, which is exactly the
    bonafide fraction of the validation set.

  * The EER routine DOES compute the equal-error threshold, but
    Evaluator.evaluate filters it out before returning:

        if "threshold" not in k.lower():
            results[k] = v

    so the one number you would need to fix the other two never
    reaches you.

There is no precision or recall metric at all.

EER itself is fine, because it is threshold-free. Everything else has
to be computed here, from the per-sample scores `deepfense test` writes
to results/predictions/<dataset>_predictions.txt.

CALIBRATION -- the part that matters

A threshold-dependent metric is meaningless without saying where the
threshold came from, and choosing it on the same data you report is
circular: you would be tuning on test and reporting the result as if it
generalised.

So:

    --calibrate FILE   pick the threshold at the EER point of THIS
                       file, which should be validation
    --evaluate  FILE   apply that threshold to THIS file, the test set

If you omit --calibrate, the threshold is taken from the evaluation
file itself. That is an ORACLE number: it is the best the model could
have done with perfect threshold knowledge, it is optimistic, and the
report labels it as such. Use it for a sanity check, never for a
headline.

CONFIDENCE INTERVALS

EER is reported with a bootstrap percentile interval. With a few
thousand trials, the difference between 3.0% and 3.4% is usually inside
the noise, and a thesis that reports a bare point estimate invites the
question of whether the ablation differences are real.

Usage:

    # honest: threshold from validation, metrics on test
    python training\score_predictions.py --calibrate outputs\RUN\results\predictions\val_predictions.txt --evaluate outputs\RUN\results\predictions\test_predictions.txt

    # several test conditions against one calibration
    python training\score_predictions.py --calibrate val_predictions.txt --evaluate clean.txt --evaluate opus.txt --evaluate amr.txt

    # oracle threshold, clearly labelled, for a quick look
    python training\score_predictions.py --evaluate test_predictions.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 1 = bonafide, 0 = spoof, matching label_map in the training configs.
BONAFIDE = 1


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------


def load_predictions(path):
    """
    Read a deepfense predictions file.

    Format written by cli/commands/test.py:

        ID_audio,label,score_class0,score_class1

    Returns (ids, labels, scores) where `scores` is the log-odds-style
    difference score_class1 - score_class0, so that larger means more
    bonafide. That matches the convention the trainer uses internally
    (`scores[:, 1] - scores[:, 0]`).
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"{p} not found")

    ids, labels, s0, s1 = [], [], [], []

    with p.open(encoding="utf-8") as fh:
        header = fh.readline().strip()

        if not header.lower().startswith("id_audio"):
            raise ValueError(
                f"{p}: unexpected header {header!r}. Expected the "
                "deepfense predictions format "
                "'ID_audio,label,score_class0,score_class1'."
            )

        for n, line in enumerate(fh, 2):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 4:
                raise ValueError(
                    f"{p}:{n}: expected 4 fields, got {len(parts)}"
                )

            ids.append(",".join(parts[:-3]))
            labels.append(int(parts[-3]))
            s0.append(float(parts[-2]))
            s1.append(float(parts[-1]))

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(s1, dtype=float) - np.asarray(s0, dtype=float)

    if labels.size == 0:
        raise ValueError(f"{p}: no rows")

    present = set(np.unique(labels))

    if present != {0, 1}:
        raise ValueError(
            f"{p}: need both classes to score, found labels {sorted(present)}"
        )

    return np.array(ids), labels, scores


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------


def det_curve(labels, scores):
    """False-reject and false-accept rates over every threshold."""
    order = np.argsort(scores, kind="mergesort")
    lab = labels[order]
    thr = scores[order]

    n_bona = int((lab == BONAFIDE).sum())
    n_spoof = int((lab != BONAFIDE).sum())

    # Sweeping the threshold upward: bonafide below it are rejected,
    # spoof above it are accepted.
    frr = np.cumsum(lab == BONAFIDE) / n_bona
    far = 1.0 - (np.cumsum(lab != BONAFIDE) / n_spoof)

    return frr, far, thr


def compute_eer(labels, scores):
    """EER and the threshold at which it occurs."""
    frr, far, thr = det_curve(labels, scores)
    idx = int(np.nanargmin(np.abs(frr - far)))

    return float((frr[idx] + far[idx]) / 2.0), float(thr[idx])


def compute_auc(labels, scores):
    """AUC via the rank-sum identity -- no interpolation, exact."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    pos = labels == BONAFIDE
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())

    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def confusion(labels, scores, threshold):
    """
    Counts at a fixed threshold, with bonafide as the positive class.

    Predicted bonafide when score >= threshold.
    """
    pred = scores >= threshold
    actual = labels == BONAFIDE

    return {
        "tp": int(np.sum(pred & actual)),
        "fp": int(np.sum(pred & ~actual)),
        "tn": int(np.sum(~pred & ~actual)),
        "fn": int(np.sum(~pred & actual)),
    }


def prf(cm):
    """Accuracy, precision, recall, F1 -- and their spoof-side duals."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    total = tp + fp + tn + fn

    def safe(num, den):
        return float(num / den) if den else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    spoof_precision = safe(tn, tn + fn)
    spoof_recall = safe(tn, tn + fp)

    return {
        "accuracy": safe(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": safe(2 * precision * recall, precision + recall),
        "spoof_precision": spoof_precision,
        "spoof_recall": spoof_recall,
        "spoof_f1": safe(
            2 * spoof_precision * spoof_recall, spoof_precision + spoof_recall
        ),
        "macro_f1": 0.0,  # filled below
    }


def bootstrap_eer(labels, scores, n_boot, seed):
    """
    Percentile interval for EER by resampling clips with replacement.

    Resamples within each class so the class balance is preserved --
    otherwise the interval widens for a reason that has nothing to do
    with the model.
    """
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(labels == BONAFIDE)
    idx_neg = np.flatnonzero(labels != BONAFIDE)
    out = []

    for _ in range(n_boot):
        take = np.concatenate(
            [
                rng.choice(idx_pos, idx_pos.size, replace=True),
                rng.choice(idx_neg, idx_neg.size, replace=True),
            ]
        )
        eer, _ = compute_eer(labels[take], scores[take])
        out.append(eer)

    out = np.asarray(out)

    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------


def report(name, labels, scores, threshold, thr_source, args):
    eer, own_thr = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    cm = confusion(labels, scores, threshold)
    m = prf(cm)
    m["macro_f1"] = (m["f1"] + m["spoof_f1"]) / 2.0

    n_bona = int((labels == BONAFIDE).sum())
    n_spoof = int(len(labels) - n_bona)

    print("=" * 74)
    print(f"  {name}")
    print("=" * 74)
    print(
        f"  clips            : {len(labels)}  "
        f"(bonafide {n_bona}, spoof {n_spoof})"
    )
    print()
    print("  THRESHOLD-FREE")
    print(f"    EER            : {eer:.4f}")

    if args.bootstrap:
        lo, hi = bootstrap_eer(labels, scores, args.bootstrap, args.seed)
        print(
            f"    EER 95% CI     : [{lo:.4f}, {hi:.4f}]"
            f"   ({args.bootstrap} bootstrap resamples)"
        )

    print(f"    AUC            : {auc:.4f}")
    print()
    print(f"  AT THRESHOLD {threshold:+.6f}   ({thr_source})")
    print(f"    accuracy       : {m['accuracy']:.4f}")
    print()
    print(f"    {'':<10} {'precision':>10} {'recall':>10} {'F1':>10}")
    print(
        f"    {'bonafide':<10} {m['precision']:>10.4f}"
        f" {m['recall']:>10.4f} {m['f1']:>10.4f}"
    )
    print(
        f"    {'spoof':<10} {m['spoof_precision']:>10.4f}"
        f" {m['spoof_recall']:>10.4f} {m['spoof_f1']:>10.4f}"
    )
    print(f"    {'macro':<10} {'':>10} {'':>10} {m['macro_f1']:>10.4f}")
    print()
    print("    confusion (rows = actual, cols = predicted)")
    print(f"                  pred bonafide   pred spoof")
    print(f"      bonafide  {cm['tp']:>14} {cm['fn']:>12}")
    print(f"      spoof     {cm['fp']:>14} {cm['tn']:>12}")

    if thr_source.startswith("ORACLE"):
        print()
        print("    WARNING: this threshold was chosen on THIS data, so")
        print("    every threshold-dependent number above is optimistic.")
        print("    Pass --calibrate with your validation predictions for")
        print("    a number you can report.")
    elif abs(threshold - own_thr) > 1e-9:
        print()
        print(f"    (this split's own EER threshold would be {own_thr:+.6f};")
        print("     the gap is the calibration mismatch, which is real and")
        print("     is exactly what reporting an honest threshold costs)")

    print()

    return {"name": name, "eer": eer, "auc": auc, **m, **cm}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy / precision / recall / F1 from deepfense "
            "prediction files, at a properly calibrated threshold."
        )
    )

    parser.add_argument(
        "--evaluate",
        action="append",
        required=True,
        metavar="FILE",
        help="predictions file to score; repeatable for several conditions",
    )
    parser.add_argument(
        "--calibrate",
        metavar="FILE",
        help=(
            "predictions file to take the threshold from -- should be "
            "VALIDATION. Omit for an oracle threshold, which the report "
            "labels as optimistic."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help="bootstrap resamples for the EER interval (0 to skip)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="also write one row per evaluated file to this CSV",
    )

    args = parser.parse_args()

    threshold = None
    thr_source = None

    if args.calibrate:
        try:
            _, cal_labels, cal_scores = load_predictions(args.calibrate)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        cal_eer, threshold = compute_eer(cal_labels, cal_scores)
        thr_source = f"EER point of {Path(args.calibrate).name}"

        print()
        print(f"Calibrated on {args.calibrate}")
        print(
            f"  {len(cal_labels)} clips, EER {cal_eer:.4f}, "
            f"threshold {threshold:+.6f}"
        )
        print()

    rows = []

    for path in args.evaluate:
        try:
            _, labels, scores = load_predictions(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        if threshold is None:
            _, thr = compute_eer(labels, scores)
            rows.append(
                report(
                    path, labels, scores, thr, "ORACLE, self-calibrated", args
                )
            )
        else:
            rows.append(
                report(path, labels, scores, threshold, thr_source, args)
            )

    if args.csv and rows:
        import csv

        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        print(f"Wrote {args.csv}")

    if len(rows) > 1:
        print("=" * 74)
        print("  SUMMARY")
        print("=" * 74)
        print(f"  {'condition':<34} {'EER':>8} {'F1':>8} {'acc':>8}")

        for r in rows:
            print(
                f"  {Path(r['name']).name:<34} {r['eer']:>8.4f}"
                f" {r['macro_f1']:>8.4f} {r['accuracy']:>8.4f}"
            )

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
