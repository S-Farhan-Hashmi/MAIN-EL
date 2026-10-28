#!/usr/bin/env python3
"""
Preprocess the lake tables produced by build_lake_tables.py.

FIT on the training file (learns which columns to keep and the median values),
then APPLY the same settings to the test file so nothing leaks from test to train.

  python preprocess_lake.py --in out/lake_training.csv --out out/lake_clean_train.csv --params out/preprocess_params.json
  python preprocess_lake.py --in out/lake_test.csv     --out out/lake_clean_test.csv  --params out/preprocess_params.json --apply

Options (fit mode only)
  --max_missing 0.30   drop a parameter column if more than 30% of rows are blank
  --keep_2023          keep the July 2023 month (default: excluded, see report)
  --keep_no_coords     keep lakes with no coordinates (default: dropped, they have no population)
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

PARAMS = ["temperature_c", "do_mgl", "ph", "conductivity", "bod_mgl", "nitrate_n", "nitrite_n",
          "fecal_coliform", "total_coliform", "turbidity_ntu", "cod_mgl", "tkn", "ammonia_n",
          "chloride", "tds", "tss", "total_phosphate", "ortho_phosphate"]
TARGETS = ["bod_mgl", "do_mgl"]                     # kept in raw units (a log copy of BOD is added)
SKEWED = ["conductivity", "nitrate_n", "nitrite_n", "fecal_coliform", "total_coliform",
          "turbidity_ntu", "cod_mgl", "tkn", "ammonia_n", "chloride", "tds", "tss",
          "total_phosphate", "ortho_phosphate"]     # replaced by log10 versions

# physically impossible values -> blank (then imputed)
VALID = {"temperature_c": (15, 40), "ph": (0, 14), "do_mgl": (0, 20),
         "fecal_coliform": (0, 1e8), "total_coliform": (0, 1e8)}


def clean_use_class(s):
    t = str(s).lower()
    if "fisher" in t or "wild" in t:
        return "D"
    if "irrig" in t or "cooling" in t:
        return "E"
    m = re.match(r"^\W*([a-e])\b", str(s).strip().lower())
    return m.group(1).upper() if m else np.nan


def log10_safe(x):
    return np.log10(x.clip(lower=1e-3))


def report_structural_gaps(d, log):
    """Columns that are 100% blank for a whole monthly file point to a PDF parsing problem."""
    if "source_file" not in d.columns:
        return
    gap = d.groupby("source_file")[PARAMS].apply(lambda g: g.isna().all())
    lines = []
    for f, row in gap.iterrows():
        cols = [c for c in PARAMS if row[c]]
        if cols:
            lines.append(f"  {f}: completely blank -> {', '.join(cols)}")
    if lines:
        log.append("WARNING: columns blank for an entire monthly file (likely a PDF parsing gap, "
                   "or a column that file never had):")
        log.extend(lines)


def basic_cleaning(d, log):
    n0 = len(d)
    river = d["lake_id"].str.startswith("river")
    amb = d["lake_id"].str.endswith("_ambiguous") | d["coord_source"].eq("ambiguous")
    junk = d["lake_id"].str.len() < 6
    log.append(f"dropped river stations: {int(river.sum())}, ambiguous-name rows: {int(amb.sum())}, "
               f"junk ids: {int(junk.sum())}")
    d = d[~(river | amb | junk)].copy()

    d["use_class"] = d["use_class"].map(clean_use_class)
    d["is_class_E"] = d["use_class"].eq("E").astype(int)

    for col, (lo, hi) in VALID.items():
        bad = d[col].notna() & ~d[col].between(lo, hi)
        if bad.any():
            log.append(f"set {int(bad.sum())} impossible {col} value(s) to blank (outside {lo}-{hi})")
        d.loc[bad, col] = np.nan

    dup = d.duplicated(["lake_id", "date"], keep="first")
    if dup.any():
        log.append(f"dropped {int(dup.sum())} duplicate lake/month rows")
        d = d[~dup]
    log.append(f"rows after basic cleaning: {len(d)} (from {n0})")
    return d


def build_features(d, keep_params, medians, flag_cols):
    out = pd.DataFrame({"date": d["date"], "lake_id": d["lake_id"], "year": d["year"], "month": d["month"]})
    out["lat"], out["lon"] = d["lat"], d["lon"]
    # NOTE: the KSPCB use-based class is NOT a feature. It is assigned from that month's measured
    # DO/BOD (class E whenever DO < 3, never when DO >= 4), so it would leak the target.
    out["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    out["log_pop_1km"] = np.log10(d["pop_1km"] + 1)
    out["log_pop_2km"] = np.log10(d["pop_2km"] + 1)

    for p in keep_params:
        x = d[p]
        if p in flag_cols:
            out[p + "_was_missing"] = x.isna().astype(int)
        x = x.fillna(medians[p])
        if p in SKEWED:
            out["log_" + p] = log10_safe(x)
        else:
            out[p] = x
    if "bod_mgl" in keep_params:
        out["log_bod_mgl"] = log10_safe(out["bod_mgl"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--apply", action="store_true", help="use saved params (for the test file)")
    ap.add_argument("--max_missing", type=float, default=0.30)
    ap.add_argument("--keep_2023", action="store_true")
    ap.add_argument("--keep_no_coords", action="store_true")
    a = ap.parse_args()

    log = []
    d = pd.read_csv(a.inp, parse_dates=["date"])
    log.append(f"loaded {len(d)} rows, {d.lake_id.nunique()} lakes from {a.inp}")

    if a.apply:
        P = json.load(open(a.params))
        keep_2023, keep_no_coords = P["keep_2023"], P["keep_no_coords"]
    else:
        keep_2023, keep_no_coords = a.keep_2023, a.keep_no_coords
        report_structural_gaps(d, log)

    d = basic_cleaning(d, log)

    if not keep_2023:
        m = d["date"].dt.year.eq(2023)
        if m.any():
            log.append(f"excluded July 2023 ({int(m.sum())} rows): not continuous with the rest and it "
                       "lacks temperature, nitrite, TKN, TSS and both phosphates")
        d = d[~m]
    if not keep_no_coords:
        m = d["pop_1km"].isna()
        log.append(f"dropped {int(m.sum())} rows with no coordinates/population ({d[m].lake_id.nunique()} lakes)")
        d = d[~m]

    if a.apply:
        keep_params, medians, flag_cols = P["keep_params"], P["medians"], P["flag_cols"]
    else:
        miss = d[PARAMS].isna().mean()
        keep_params = [p for p in PARAMS if miss[p] <= a.max_missing or p in TARGETS]
        dropped = [f"{p} ({miss[p]:.0%})" for p in PARAMS if p not in keep_params]
        log.append("parameter missing % after cleaning: " +
                   ", ".join(f"{p} {miss[p]:.0%}" for p in PARAMS))
        log.append(f"dropped columns (> {a.max_missing:.0%} blank): {', '.join(dropped) or 'none'}")
        medians = {p: float(d[p].median()) for p in keep_params}
        flag_cols = [p for p in keep_params if miss[p] >= 0.05]
        json.dump({"keep_params": keep_params, "medians": medians, "flag_cols": flag_cols,
                   "keep_2023": keep_2023, "keep_no_coords": keep_no_coords,
                   "max_missing": a.max_missing}, open(a.params, "w"), indent=2)

    out = build_features(d, keep_params, medians, flag_cols)
    out.to_csv(a.out, index=False)

    log.append(f"wrote {a.out}: {out.shape[0]} rows x {out.shape[1]} columns, {out.lake_id.nunique()} lakes")
    log.append(f"months: {', '.join(sorted(out.date.dt.strftime('%Y-%m').unique()))}")
    log.append(f"columns: {', '.join(out.columns)}")
    text = "\n".join(log)
    print(text)
    with open(os.path.splitext(a.out)[0] + "_report.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
