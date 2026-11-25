"""Build the recomputed EC test inputs (paper, Section IV-B).

The EC models are trained on MAEC (MAEC-15 + MAEC-16 training splits), whose historical-volatility features
past_1 ... past_29 are defined as

    v_j = ln sqrt( 1/(30-j) * sum_{t=d-29}^{d-j} (r_t - m)^2 ),   j = 1..29,   m = mean(r_{d-29}, ..., r_d),

with r_t = p_t / p_{t-1} - 1 the daily return of adjusted closes and d the call date. The features released with
the EC test set use a different definition, so for EC we recompute v_1 ... v_29 from adjusted closing prices with
the MAEC definition. The EC targets (future_3, future_7, future_15, future_30) are the released ones, unchanged.

Inputs
  --ec-test     KeFVP's EC test file, test_split_Avg_Series_WITH_LOG.csv (downloaded automatically if omitted,
                from github.com/hankniu01/KeFVP at commit 95993892)
  --prices      data/ec/ec_adjusted_closes.csv              daily adjusted closes, one column per ticker
  --supplement  data/ec/ec_adjusted_closes_supplement.csv   Yahoo Finance adjusted closes for six calls the
                                                            first file cannot serve (ABC, BLL, cbre, DG, PKI, FB)
Output
  --out         data/ec/ec_test_corrected.csv               110 EC test calls, columns ticker, name, year, month,
                                                            day, text_file_name, future_3/7/15/30, past_1..past_29

Ticker matching: the EC ticker as written, else its upper-case form, else the alias RHI.F -> RHT (Red Hat).
Two of the 112 EC test calls have no usable price history (CBS, WRK) and are left out.

usage: python scripts/build_ec_features.py [--out data/ec/ec_test_corrected.csv]
"""
import argparse
import math
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
KEFVP_EC_URL = ("https://raw.githubusercontent.com/hankniu01/KeFVP/95993892d087599dc5530804aada73f7b5004f39/"
                "price_data/test_split_Avg_Series_WITH_LOG.csv")
ALIASES = {"RHI.F": "RHT"}
STEPS = range(1, 30)


def load_prices(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def returns(prices, ticker):
    p = prices[ticker].to_numpy(dtype=np.float64)
    r = np.full_like(p, np.nan)
    r[1:] = p[1:] / p[:-1] - 1.0          # a missing close makes the neighbouring returns missing
    return r


def maec_features(r, d):
    """v_1..v_29 for a call at trading-day position d (needs closes from d-30 to d)."""
    m = r[d - 29:d + 1].mean()
    return [math.log(math.sqrt(float(np.mean((r[d - 29:d - j + 1] - m) ** 2)))) for j in STEPS]


def features_from(prices, ticker, date):
    """Feature list, or None if the price file cannot serve this call."""
    if ticker is None or date not in prices.index:
        return None
    d = prices.index.get_loc(date)
    r = returns(prices, ticker)
    if d - 29 < 1 or np.isnan(r[d - 29:d + 1]).any():
        return None
    return maec_features(r, d)


def match(ticker, columns):
    for t in (ticker, str(ticker).upper(), ALIASES.get(ticker)):
        if t in columns:
            return t
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ec-test", help="KeFVP test_split_Avg_Series_WITH_LOG.csv (downloaded if omitted)")
    ap.add_argument("--prices", default=os.path.join(ROOT, "data", "ec", "ec_adjusted_closes.csv"))
    ap.add_argument("--supplement", default=os.path.join(ROOT, "data", "ec", "ec_adjusted_closes_supplement.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ec", "ec_test_corrected.csv"))
    a = ap.parse_args()

    ec = pd.read_csv(a.ec_test or urllib.request.urlopen(KEFVP_EC_URL))
    prices, supplement = load_prices(a.prices), load_prices(a.supplement)

    rows, skipped = [], []
    for _, call in ec.iterrows():
        date = pd.Timestamp(year=int(call["year"]), month=int(call["month"]), day=int(call["day"]))
        feats = features_from(prices, match(call["ticker"], prices.columns), date)
        if feats is None and call["ticker"] in supplement.columns:
            feats = features_from(supplement, call["ticker"], date)
        if feats is None:
            skipped.append(f"{call['ticker']} {date.date()}")
            continue
        row = {c: call[c] for c in ("ticker", "name", "year", "month", "day", "text_file_name")}
        row.update({f"future_{h}": call[f"future_{h}"] for h in (3, 7, 15, 30)})
        row.update({f"past_{j}": v for j, v in zip(STEPS, feats)})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, float_format="%.17g", lineterminator="\n")
    print(f"{len(out)} of {len(ec)} EC test calls written to {os.path.normpath(a.out)}")
    print("left out (no usable price history):", ", ".join(skipped) or "none")


if __name__ == "__main__":
    main()
