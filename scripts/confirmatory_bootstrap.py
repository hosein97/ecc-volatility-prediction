"""Confirmatory analysis of the MAEC-16 ablation (paper, Section V-F), from the released predictions.

The pre-specified test compares the proposed model (Past Volatility + QA Text + Audio, Volatility Prompt) with
five variants on the MAEC-16 test calls that every one of them scores (the 272 calls with generated QA units):

  * Per-call loss: for each call and horizon, the mean over the 55 seeds of the squared error; then the mean over
    the four horizons.
  * Difference: comparator minus proposed, so a positive value means the proposed model has the lower error.
  * Paired bootstrap over calls: 5,000 resamples, numpy.random.default_rng(0), one index matrix shared by all
    comparisons, calls in sorted "ticker|date" order; percentile 95% intervals; two-sided p-value by the shift
    method, p = (1 + #{|b - d| >= |d|}) / (B + 1).
  * Holm correction across the five comparisons, alpha = 0.05.

Per-horizon differences and intervals are printed as well; they are descriptive (no p-value, no correction).

usage:
    python scripts/confirmatory_bootstrap.py                          # uses results/predictions/
    python scripts/confirmatory_bootstrap.py --maec16 B.csv --out results/tables
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402

N_BOOT = 5000
BOOT_SEED = 0
ALPHA = 0.05
COMPARATORS = [  # (label, modality, fusion_type), in the paper's order
    ("Past Volatility", "vol_only", "n/a"),
    ("Past Volatility + QA Text", "text_only", "residual_gated"),
    ("Past Volatility + QA Audio", "audio_only", "residual_gated"),
    ("Past Volatility + QA Text + Audio (General Prompt)", "text_audio_prompt1", "residual_gated"),
    ("Past Volatility + Transcript Chunks + Audio", "chunk_text_audio", "residual_gated"),
]
COLS = ["avg"] + [f"{h}d" for h in mt.HORIZONS]


def per_call_loss(df, model):
    """call x {avg, 3d, 7d, 15d, 30d}: mean squared error over seeds, per horizon and averaged over horizons."""
    m = df[(df.modality == model[0]) & (df.fusion_type == model[1])]
    m = m.assign(call=m["ticker"] + "|" + m["date"])
    t = m.groupby(["call", "horizon"])["sq_err"].mean().unstack("horizon")
    t = t[list(mt.HORIZONS)]
    t.columns = [f"{h}d" for h in mt.HORIZONS]
    t.insert(0, "avg", t.mean(axis=1))
    return t


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maec16", default=os.path.join(mt.DEFAULT_PRED, "maec16_predictions.csv.gz"))
    ap.add_argument("--out", default=os.path.join(mt.HERE, "..", "results", "tables"))
    args = ap.parse_args()

    df = mt.load(args.maec16)
    proposed = per_call_loss(df, mt.PROPOSED)
    losses = {label: per_call_loss(df, (mod, fus)) for label, mod, fus in COMPARATORS}

    # common calls: those every compared model has its own prediction for (no fallback is used here)
    calls = proposed.index
    for t in losses.values():
        calls = calls.intersection(t.index)
    calls = calls.sort_values()
    n = len(calls)
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, (N_BOOT, n))

    rows = []
    for label, t in losses.items():
        for col in COLS:
            d = (t.loc[calls, col] - proposed.loc[calls, col]).to_numpy()
            obs = d.mean()
            b = d[idx].mean(axis=1)
            rows.append({
                "comparator": label, "metric": col, "n_calls": n,
                "diff_comparator_minus_proposed": obs,
                "ci95_low": float(np.percentile(b, 2.5)), "ci95_high": float(np.percentile(b, 97.5)),
                "p_two_sided": (1 + np.sum(np.abs(b - obs) >= abs(obs))) / (N_BOOT + 1) if col == "avg" else np.nan,
            })
    res = pd.DataFrame(rows)
    primary = res[res.metric == "avg"].copy()
    primary["p_holm"] = holm(primary["p_two_sided"])
    primary["significant_holm_0.05"] = primary["p_holm"] < ALPHA
    per_horizon = res[res.metric != "avg"].drop(columns="p_two_sided")

    print(f"Confirmatory analysis, MAEC-16, {n} common calls ({N_BOOT} resamples, default_rng({BOOT_SEED}))")
    print("Difference = comparator minus proposed (positive: the proposed model has the lower average MSE)\n")
    for _, r in primary.iterrows():
        print(f"  {r.comparator:<52} {r.diff_comparator_minus_proposed:+.4f}  "
              f"[{r.ci95_low:+.5f}, {r.ci95_high:+.5f}]  p = {r.p_two_sided:.4f}  p_Holm = {r.p_holm:.4f}")
    print("\nPer horizon (descriptive):")
    for _, r in per_horizon.iterrows():
        print(f"  {r.comparator:<52} {r.metric:>4} {r.diff_comparator_minus_proposed:+.4f}  "
              f"[{r.ci95_low:+.5f}, {r.ci95_high:+.5f}]")

    os.makedirs(args.out, exist_ok=True)
    primary.drop(columns="metric").to_csv(os.path.join(args.out, "maec16_confirmatory.csv"), index=False)
    per_horizon.to_csv(os.path.join(args.out, "maec16_confirmatory_per_horizon.csv"), index=False)
    print(f"\nwritten: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
