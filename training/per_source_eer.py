r"""
per_source_eer.py

Per-spoof-system EER from a saved prediction file, without re-running
the model.

Usage:

    python training\per_source_eer.py ^
        --predictions outputs\RUN\results\predictions\test_opus_predictions.txt

    # compare several conditions side by side
    python training\per_source_eer.py ^
        --predictions outputs\RUN\results\predictions\voice-conv_predictions.txt ^
        --predictions outputs\RUN\results\predictions\test_opus_predictions.txt ^
        --predictions outputs\RUN\results\predictions\test_amr_predictions.txt
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Missing dependency. Run: pip install pandas pyarrow numpy",
          file=sys.stderr)
    sys.exit(1)


def eer(labels, scores) -> float:
    """
    Equal error rate with bonafide (1) as the target class.

    Threshold-free: sweeps every operating point and returns the one
    where the two error rates meet.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    y = labels[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    frr = np.cumsum(y) / n_pos            # bonafide rejected below thr
    far = 1.0 - np.cumsum(1 - y) / n_neg  # spoof accepted above thr
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0)


def load_manifest(splits_dir: Path) -> pd.DataFrame:
    parts = []
    for s in ("train", "val", "test"):
        f = splits_dir / f"{s}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f)[["ID", "spoof_method"]])
    if not parts:
        print(f"ERROR: no parquets under {splits_dir}", file=sys.stderr)
        sys.exit(1)
    return pd.concat(parts, ignore_index=True).drop_duplicates("ID")


def report(pred_path: Path, man: pd.DataFrame) -> dict:
    p = pd.read_csv(pred_path)
    need = {"ID_audio", "label", "score_class1"}
    if not need <= set(p.columns):
        print(f"ERROR: {pred_path} lacks {need - set(p.columns)}",
              file=sys.stderr)
        sys.exit(1)

    d = p.merge(man, left_on="ID_audio", right_on="ID", how="left")
    unmatched = d["spoof_method"].isna().sum()

    # Sanity: score_class1 should be HIGHER for bonafide (label 1).
    mb = d.loc[d.label == 1, "score_class1"].mean()
    ms = d.loc[d.label == 0, "score_class1"].mean()
    if not mb > ms:
        print(f"  WARNING: score_class1 is not higher for bonafide "
              f"({mb:.3f} vs {ms:.3f}) -- polarity may be inverted")

    bona = d[d.label == 1]
    out = {"pooled": eer(d.label, d.score_class1)}
    print(f"\n  {pred_path.name}")
    print(f"    {len(d)} clips   {len(bona)} bonafide   "
          f"{unmatched} unmatched in the manifest")
    print(f"    {'pooled':22s} {out['pooled']:.4f}")
    for m in sorted(x for x in d["spoof_method"].dropna().unique()
                    if x != "none"):
        sub = pd.concat([bona, d[d.spoof_method == m]])
        out[m] = eer(sub.label, sub.score_class1)
        print(f"    {m:22s} {out[m]:.4f}   ({(d.spoof_method == m).sum()} "
              f"spoof clips vs the shared bonafide pool)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-spoof-system EER from saved predictions.")
    ap.add_argument("--predictions", action="append", required=True,
                    help="prediction .txt; repeat to compare conditions")
    ap.add_argument("--splits-dir", default="manifests/splits_vc")
    args = ap.parse_args()

    man = load_manifest(Path(args.splits_dir))
    print(f"\n  manifest: {args.splits_dir}  ({len(man)} ids)")

    rows = {}
    for f in args.predictions:
        rows[Path(f).stem] = report(Path(f), man)

    if len(rows) > 1:
        keys = sorted({k for r in rows.values() for k in r})
        w = max(len(n) for n in rows) + 2
        print(f"\n  {'condition':{w}s}" + "".join(f"{k:>20s}" for k in keys))
        print("  " + "-" * (w + 20 * len(keys)))
        for n, r in rows.items():
            print(f"  {n:{w}s}" + "".join(
                f"{r.get(k, float('nan')):>20.4f}" for k in keys))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
