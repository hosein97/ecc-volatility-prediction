"""Secondary statistics reported in the paper's text (Sections V-B and V-F), from the released predictions.

make_tables.py builds our rows of Table I and Table III. This script computes further numbers quoted in the text:

  * Common-call sensitivity analysis (V-B): every model scored only on the calls with generated QA units
    (EC: 109, MAEC-15: 150, MAEC-16: 272), so that no variant relies on the Past Volatility fallback.
    Mean MSE over seeds for all rows, and whether every Table I / Table III significance mark is the same
    as on all test calls.
  * Pairwise seed-paired t-tests on MAEC-16, on all test calls and on the common calls (V-F, RQ2/RQ3):
      - QA Text vs QA Audio, two-sided
      - Transcript Chunks + Audio vs Proposed, two-sided
      - Proposed (Volatility Prompt) vs General Prompt, one-sided (Proposed lower)
      - Proposed vs QA Text and vs QA Audio, one-sided (Proposed lower)
  * Reduction in MSE from Past Volatility to Proposed per horizon on MAEC-16, and the spread of average MSE
    over QA Text, QA Audio, both prompts and Transcript Chunks + Audio.

The evaluation protocol is that of make_tables.py (per seed: MSE over calls per horizon, Avg = mean of the four
horizons; reported value: mean over the 55 seeds; scipy.stats.ttest_rel over seeds).

usage:
    python scripts/sensitivity_and_twosided.py                      # uses results/predictions/
    python scripts/sensitivity_and_twosided.py --maec15 A.csv --maec16 B.csv --ec C.csv --out results/tables
"""
import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402

LABEL = {label: (mod, fus) for label, mod, fus in mt.TABLE_III}
BASE_L, TEXT, AUDIO, GENERAL, PROPOSED_L, CHUNKS, CONCAT, QA_ONLY = (label for label, _, _ in mt.TABLE_III)
DATASETS = (("EC", "ec"), ("MAEC-15", "maec15"), ("MAEC-16", "maec16"))


def common_calls(df):
    """Keep only the calls on which the proposed model has its own prediction (the calls with QA units)."""
    calls = df[(df.modality == mt.PROPOSED[0]) & (df.fusion_type == mt.PROPOSED[1])][["ticker", "date"]]
    return df.merge(calls.drop_duplicates(), on=["ticker", "date"])


def ttest(a, b, alternative):
    seeds = a.index.intersection(b.index)
    if len(seeds) != len(a) or len(seeds) != len(b):
        raise ValueError("the two models were not run with the same seeds")
    return {m: stats.ttest_rel(a.loc[seeds, m], b.loc[seeds, m], alternative=alternative).pvalue for m in mt.METRICS}


def population_rows(frames, population):
    """Mean MSEs and marks of Tables I and III for one population ('all' or 'common')."""
    rows = []
    for name, df in frames.items():
        models = [(BASE_L, mt.BASE), (PROPOSED_L, mt.PROPOSED)] if name != "MAEC-16" else list(LABEL.items())
        mse = {label: mt.per_seed_mse(df, model) for label, model in models}
        for label, m in mse.items():
            p = None if label == PROPOSED_L else ttest(mse[PROPOSED_L], m, "less")
            row = {"population": population, "dataset": name, "model": label, "calls": df[["ticker", "date"]]
                   .drop_duplicates().shape[0]}
            for k in mt.METRICS:
                row[k] = m[k].mean()
                row[f"{k}_mark"] = "" if p is None else mt.stars(p[k])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maec15", default=os.path.join(mt.DEFAULT_PRED, "maec15_predictions.csv.gz"))
    ap.add_argument("--maec16", default=os.path.join(mt.DEFAULT_PRED, "maec16_predictions.csv.gz"))
    ap.add_argument("--ec", default=os.path.join(mt.DEFAULT_PRED, "ec_predictions.csv.gz"))
    ap.add_argument("--out", default=os.path.join(mt.HERE, "..", "results", "tables"))
    a = ap.parse_args()

    raw = {}
    for name, key in DATASETS:
        df = mt.load(getattr(a, key))
        raw[name] = df[df.dataset == key]
    all_calls = {name: mt.with_base_fallback(df) for name, df in raw.items()}
    common = {name: common_calls(df) for name, df in raw.items()}

    # 1. common-call sensitivity analysis
    pops = pd.DataFrame(population_rows(all_calls, "all") + population_rows(common, "common"))
    marks = [f"{k}_mark" for k in mt.METRICS]
    a_m = pops[pops.population == "all"].set_index(["dataset", "model"])[marks]
    c_m = pops[pops.population == "common"].set_index(["dataset", "model"])[marks]
    same_marks = a_m.equals(c_m.loc[a_m.index])

    # 2. pairwise tests on MAEC-16
    tests = []
    for population, frames in (("all", all_calls), ("common", common)):
        mse = {label: mt.per_seed_mse(frames["MAEC-16"], model) for label, model in LABEL.items()}
        for x, y, alt in ((TEXT, AUDIO, "two-sided"), (CHUNKS, PROPOSED_L, "two-sided"),
                          (PROPOSED_L, GENERAL, "less"), (PROPOSED_L, TEXT, "less"), (PROPOSED_L, AUDIO, "less")):
            p = ttest(mse[x], mse[y], alt)
            tests.append({"population": population, "a": x, "b": y,
                          "alternative": "two-sided" if alt == "two-sided" else "one-sided (a lower)",
                          **{f"{k}_p": p[k] for k in mt.METRICS}, "min_p": min(p.values())})
    tests = pd.DataFrame(tests)

    # 3. MAEC-16 gains and spread (all test calls)
    mse = {label: mt.per_seed_mse(all_calls["MAEC-16"], model) for label, model in LABEL.items()}
    gain = {k: mse[BASE_L][k].mean() - mse[PROPOSED_L][k].mean() for k in mt.METRICS}
    group = (TEXT, AUDIO, GENERAL, PROPOSED_L, CHUNKS)
    spread = max(mse[g]["avg"].mean() for g in group) - min(mse[g]["avg"].mean() for g in group)
    over = {b: mse[b]["avg"].mean() - mse[PROPOSED_L]["avg"].mean() for b in (TEXT, AUDIO)}

    os.makedirs(a.out, exist_ok=True)
    pops.to_csv(os.path.join(a.out, "populations_all_vs_common.csv"), index=False)
    tests.to_csv(os.path.join(a.out, "maec16_pairwise_ttests.csv"), index=False)

    print("Common-call sensitivity analysis (mean MSE over seeds; marks against the proposed model)\n")
    for _, r in pops[pops.population == "common"].iterrows():
        cells = "".join(f"{r[k]:>10.4f}{r[k + '_mark']:<2}" for k in mt.METRICS)
        print(f"{r['dataset']:<8}({r['calls']:>3}) {r['model']:<54}{cells}")
    print(f"\nAll significance marks the same on all test calls and on common calls: {same_marks}")

    print("\nMAEC-16 pairwise seed-paired t-tests (p per metric)\n")
    for _, r in tests.iterrows():
        ps = "  ".join(f"{k} {r[k + '_p']:.3f}" for k in mt.METRICS)
        print(f"[{r['population']:<6}] {r['a']} vs {r['b']} ({r['alternative']}): {ps}  (min {r['min_p']:.3f})")

    print("\nMAEC-16, all test calls")
    print("  MSE reduction Past Volatility -> Proposed: "
          + ", ".join(f"{k} {v:.4f}" for k, v in gain.items()))
    print(f"  Avg MSE of Proposed below QA Text by {over[TEXT]:.4f}, below QA Audio by {over[AUDIO]:.4f}")
    print(f"  spread of Avg MSE over QA Text, QA Audio, both prompts, Transcript Chunks + Audio: {spread:.4f}")
    print(f"\nwritten: {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
