r"""
speaker_bootstrap.py

Speaker-level (cluster) bootstrap for EER, with PAIRED differences.

Usage (one condition at a time, e.g. AMR-NB):

    python training\speaker_bootstrap.py ^
        --manifest manifests\splits_vc\test.parquet ^
        --pred none=outputs\RUN_NOAUG\results\predictions\test_amr_predictions.txt ^
        --pred opus=outputs\RUN_OPUS\results\predictions\test_amr_predictions.txt ^
        --pred amr=outputs\RUN_AMR\results\predictions\test_amr_predictions.txt ^
        --pred both=outputs\RUN_BOTH\results\predictions\test_amr_predictions.txt ^
        --compare opus:none --compare both:none --compare both:amr
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_predictions import compute_eer, load_predictions  # noqa: E402


def load_speakers(manifest):
    import pandas as pd

    df = pd.read_parquet(manifest, columns=["ID", "speaker_id"])
    return dict(zip(df["ID"].astype(str), df["speaker_id"].astype(str)))


def speakers_for(ids, id2spk, name):
    out, missing = [], 0
    for i in ids:
        spk = id2spk.get(i) or id2spk.get(Path(i).stem)
        if spk is None:
            missing += 1
        out.append(spk)
    if missing:
        raise ValueError(
            f"{name}: {missing} of {len(ids)} clip IDs not found in the "
            "manifest. Check that --manifest is the split these "
            "predictions were made on."
        )
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    help="name=path to a predictions file (repeatable)")
    ap.add_argument("--compare", action="append", default=[],
                    help="a:b -- report EER(a) - EER(b) (repeatable)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    id2spk = load_speakers(args.manifest)

    models = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        ids, labels, scores = load_predictions(path)
        spk = speakers_for(ids, id2spk, name)
        order = np.argsort(np.asarray(ids), kind="mergesort")
        models[name] = dict(ids=np.asarray(ids)[order], labels=labels[order],
                            scores=scores[order], spk=spk[order])

    ref = next(iter(models.values()))
    for name, m in models.items():
        if not np.array_equal(m["ids"], ref["ids"]):
            raise ValueError(
                f"{name}: clip set differs from the others; paired "
                "comparison needs every model scored on the same clips."
            )

    spk = ref["spk"]
    speakers = np.unique(spk)
    idx_by_spk = {s: np.flatnonzero(spk == s) for s in speakers}
    rng = np.random.default_rng(args.seed)

    point = {n: compute_eer(m["labels"], m["scores"])[0]
             for n, m in models.items()}
    boot = {n: np.empty(args.n_boot) for n in models}

    for b in range(args.n_boot):
        draw = rng.choice(speakers, size=len(speakers), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in draw])
        for n, m in models.items():
            lab = m["labels"][idx]
            if lab.min() == lab.max():
                boot[n][b] = np.nan
                continue
            boot[n][b] = compute_eer(lab, m["scores"][idx])[0]

    print(f"\nSpeaker-level bootstrap: {len(speakers)} speakers, "
          f"{len(spk)} clips, {args.n_boot} resamples (seed {args.seed})\n")
    print(f"  {'model':<12}{'EER %':>8}   95% CI (speaker-level)")
    for n in models:
        lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
        print(f"  {n:<12}{100 * point[n]:>8.2f}   [{100 * lo:.2f}, {100 * hi:.2f}]")

    if args.compare:
        print(f"\n  {'difference':<14}{'pts':>8}   95% CI (paired)     "
              "share of draws > 0")
        for c in args.compare:
            a, b = c.split(":")
            d = boot[a] - boot[b]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            share = np.nanmean(d > 0)
            verdict = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"  {a + ' - ' + b:<14}{100 * (point[a] - point[b]):>+8.2f}"
                  f"   [{100 * lo:+.2f}, {100 * hi:+.2f}]  {share:6.1%}  ({verdict})")
    print()


if __name__ == "__main__":
    main()