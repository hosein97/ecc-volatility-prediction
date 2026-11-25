"""Build the paper's result tables (Table I "Ours" rows and Table III) from per-call test predictions.

Input: one prediction file per dataset, in the format written by notebooks/3_volatility_prediction.ipynb
(`item_losses.csv`; columns seed, dataset, modality, fusion_type, ticker, date, horizon, pred, target, sq_err).
The files used in the paper are in results/predictions/.

Evaluation protocol (paper, Section V):
  * Every test call is scored. A QA-based model has no prediction for a call without QA units; for such a call it
    uses the same seed's Past Volatility prediction (MAEC-15: 4 calls, MAEC-16: 8, EC: 1).
  * Per seed: MSE over all test calls at each horizon (3, 7, 15, 30 days); Avg = mean of the four horizon MSEs.
  * Reported value: mean over the 55 seeds.
  * Significance marks: one-sided seed-paired t-test (scipy.stats.ttest_rel), * p < 0.05, ** p < 0.01, no correction.
      Table I:   Proposed vs Past Volatility (alternative: Proposed has the lower MSE).
      Table III: Proposed vs each other variant (alternative: Proposed has the lower MSE).

usage:
    python scripts/make_tables.py                                   # uses results/predictions/
    python scripts/make_tables.py --maec15 A.csv --maec16 B.csv --ec C.csv --out results/tables
"""
import argparse
import os

import pandas as pd
from scipy import stats

HORIZONS = (3, 7, 15, 30)
METRICS = ["avg"] + [f"{h}d" for h in HORIZONS]
BASE = ("vol_only", "n/a")
PROPOSED = ("text_audio_prompt2", "residual_gated")

# Table III rows, in the paper's order: (label, modality, fusion_type)
TABLE_III = [
    ("Past Volatility", *BASE),
    ("Past Volatility + QA Text", "text_only", "residual_gated"),
    ("Past Volatility + QA Audio", "audio_only", "residual_gated"),
    ("Past Volatility + QA Text + Audio (General Prompt)", "text_audio_prompt1", "residual_gated"),
    ("Past Volatility + QA Text + Audio (Volatility Prompt)", *PROPOSED),
    ("Past Volatility + Transcript Chunks + Audio", "chunk_text_audio", "residual_gated"),
    ("Past Volatility + QA Text + Audio (Concat)", "text_audio_prompt2", "concat"),
    ("QA Text + Audio (QA Only)", "text_audio_prompt2", "qa_only"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRED = os.path.join(HERE, "..", "results", "predictions")


def load(path):
    df = pd.read_csv(path, dtype={"ticker": str, "date": str, "modality": str, "fusion_type": str},
                     keep_default_na=False)
    return df


def with_base_fallback(df):
    """Give every model a prediction for every test call: missing calls take the same seed's Past Volatility
    prediction. Returns the completed frame."""
    base = df[(df.modality == BASE[0]) & (df.fusion_type == BASE[1])]
    if base.empty:
        raise ValueError("the Past Volatility (vol_only) predictions are missing")
    key = ["seed", "ticker", "date", "horizon"]
    parts = [base]
    for (mod, fus), g in df.groupby(["modality", "fusion_type"]):
        if (mod, fus) == BASE:
            continue
        have = g.set_index(key).index
        missing = base[~base.set_index(key).index.isin(have)].copy()
        missing["modality"], missing["fusion_type"] = mod, fus
        missing["sq_err"] = (missing["pred"] - missing["target"]) ** 2
        parts += [g, missing]
    return pd.concat(parts, ignore_index=True)


def per_seed_mse(df, model):
    """Seeds x metrics frame of MSEs for one (modality, fusion_type)."""
    g = df[(df.modality == model[0]) & (df.fusion_type == model[1])]
    if g.empty:
        raise ValueError(f"no predictions for {model}")
    m = g.groupby(["seed", "horizon"])["sq_err"].mean().unstack("horizon")
    m.columns = [f"{h}d" for h in m.columns]
    m.insert(0, "avg", m[[f"{h}d" for h in HORIZONS]].mean(axis=1))
    return m[METRICS]


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def compare(proposed, other):
    """One-sided seed-paired t-test per metric: is `proposed` lower than `other`? -> {metric: p}."""
    seeds = proposed.index.intersection(other.index)
    if len(seeds) != len(proposed) or len(seeds) != len(other):
        raise ValueError("the two models were not run with the same seeds")
    return {m: stats.ttest_rel(proposed.loc[seeds, m], other.loc[seeds, m], alternative="less").pvalue
            for m in METRICS}


def table_i(frames):
    rows = []
    for name, df in frames.items():
        base, prop = per_seed_mse(df, BASE), per_seed_mse(df, PROPOSED)
        p = compare(prop, base)
        n_calls = df[["ticker", "date"]].drop_duplicates().shape[0]
        for label, m, mark in (("Ours (Past Vol.)", base, False), ("Ours (Proposed)", prop, True)):
            row = {"dataset": name, "model": label, "calls": n_calls, "seeds": len(m)}
            for k in METRICS:
                row[k] = m[k].mean()
                row[f"{k}_p"] = p[k] if mark else float("nan")
                row[f"{k}_mark"] = stars(p[k]) if mark else ""
            rows.append(row)
    return pd.DataFrame(rows)


def table_iii(df):
    mse = {label: per_seed_mse(df, (mod, fus)) for label, mod, fus in TABLE_III}
    prop = mse[TABLE_III[4][0]]
    rows = []
    for label, _, _ in TABLE_III:
        p = None if mse[label] is prop else compare(prop, mse[label])
        row = {"variant": label, "seeds": len(mse[label])}
        for k in METRICS:
            row[k] = mse[label][k].mean()
            row[f"{k}_p"] = float("nan") if p is None else p[k]
            row[f"{k}_mark"] = "" if p is None else stars(p[k])
        rows.append(row)
    return pd.DataFrame(rows)


def show(df, label_col, digits):
    head = f"{label_col:<56}" + "".join(f"{k:>12}" for k in METRICS)
    lines = [head, "-" * len(head)]
    for _, r in df.iterrows():
        name = r[label_col] if label_col != "model" else f"{r['dataset']:<7} {r['model']}"
        lines.append(f"{name:<56}" + "".join(f"{r[k]:>{12 - 2}.{digits}f}{r[k + '_mark']:<2}" for k in METRICS))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maec15", default=os.path.join(DEFAULT_PRED, "maec15_predictions.csv.gz"))
    ap.add_argument("--maec16", default=os.path.join(DEFAULT_PRED, "maec16_predictions.csv.gz"))
    ap.add_argument("--ec", default=os.path.join(DEFAULT_PRED, "ec_predictions.csv.gz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "tables"))
    a = ap.parse_args()

    frames = {}
    for name, path in (("EC", a.ec), ("MAEC-15", a.maec15), ("MAEC-16", a.maec16)):
        df = load(path)
        df = df[df.dataset == {"EC": "ec", "MAEC-15": "maec15", "MAEC-16": "maec16"}[name]]
        frames[name] = with_base_fallback(df)

    t1, t3 = table_i(frames), table_iii(frames["MAEC-16"])
    os.makedirs(a.out, exist_ok=True)
    t1.to_csv(os.path.join(a.out, "table1_ours.csv"), index=False)
    t3.to_csv(os.path.join(a.out, "table3_ablation_maec16.csv"), index=False)

    print("Table I, our rows (MSE, mean over seeds, all test calls)\n")
    print(show(t1, "model", 3))
    print("\nTable III, MAEC-16 ablation\n")
    print(show(t3, "variant", 4))
    print(f"\nwritten: {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
