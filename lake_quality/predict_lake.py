#!/usr/bin/env python3
"""
Score a NEW lake file with the saved model (out/lake_model.joblib from train_lake_model.py).

The new file must first go through preprocess_lake.py with --apply, so it has the same columns:

  python preprocess_lake.py --in out/new_lakes.csv --out out/new_clean.csv --params out/preprocess_params.json --apply
  python predict_lake.py --model out/lake_model.joblib --in out/new_clean.csv --out_dir out

Two outputs
  predictions.csv          per row: the lake-history baseline (typical BOD, polluted share, polluted yes/no).
                           If the file contains the real BOD, it also prints accuracy against it.
  next_month_forecast.csv  per lake: forecast for the month AFTER that lake's latest month in the file
                           (gradient boosting on that month's chemistry). Needs lake_id, date, the log_* columns.

The lake-history baseline is the recommended model: on the project data it matched or beat the ML forecast.
Lakes not seen in training get the overall average (seen_in_training = 0).
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score


def baseline_predictions(d, B):
    tab = B["baseline"].set_index("lake_id")
    out = d[["lake_id", "date"]].copy()
    out["seen_in_training"] = d["lake_id"].isin(tab.index).astype(int)
    out["pred_typical_bod_mgl"] = d["lake_id"].map(tab["typical_bod_mgl"]).fillna(10 ** B["global_mean_log_bod"]).round(2)
    out["pred_polluted_share"] = d["lake_id"].map(tab["polluted_share"]).fillna(B["global_polluted_share"]).round(3)
    out["pred_polluted"] = (out["pred_polluted_share"] >= 0.5).astype(int)
    if "bod_mgl" in d.columns:
        out["actual_bod_mgl"] = d["bod_mgl"]
        out["actual_polluted"] = (d["bod_mgl"] >= B["poor_bod"]).astype(int)
    return out


def report_accuracy(out, B):
    if "actual_bod_mgl" not in out.columns:
        return
    y, p = np.log10(out["actual_bod_mgl"]), np.log10(out["pred_typical_bod_mgl"])
    print(f"\nAgainst the real BOD in this file ({len(out)} rows, {int(out.seen_in_training.sum())} lakes seen in training):")
    print(f"  log10 BOD: R2 {r2_score(y, p):.3f}, mean error {np.abs(out.actual_bod_mgl - out.pred_typical_bod_mgl).mean():.1f} mg/L")
    a, pp = out["actual_polluted"], out["pred_polluted"]
    print(f"  polluted yes/no: accuracy {accuracy_score(a, pp):.3f}, F1 {f1_score(a, pp, zero_division=0):.3f}, "
          f"share actually polluted {a.mean():.2f}", end="")
    if a.nunique() > 1:
        print(f", AUC {roc_auc_score(a, out['pred_polluted_share']):.3f}")
    else:
        print()


def next_month_forecast(d, B):
    need = set(B["lag_cols"]) | {"lake_id", "date", "log_pop_1km", "log_pop_2km", "lat", "lon"}
    if not need.issubset(d.columns):
        print("\nSkipped next-month forecast: missing columns", sorted(need - set(d.columns)))
        return None
    last = d.sort_values("date").groupby("lake_id").tail(1).copy()
    nxt = last["date"] + pd.offsets.MonthBegin(1)
    X = pd.DataFrame(index=last.index)
    for c in B["lag_cols"]:
        X["lag_" + c] = last[c]
    X["log_pop_1km"], X["log_pop_2km"] = last["log_pop_1km"], last["log_pop_2km"]
    X["month_sin"] = np.sin(2 * np.pi * nxt.dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * nxt.dt.month / 12)
    X["lat"], X["lon"] = last["lat"], last["lon"]
    X["lag_poor"] = (10 ** last["log_bod_mgl"] >= B["poor_bod"]).astype(int)
    f = pd.DataFrame({"lake_id": last["lake_id"], "latest_month": last["date"].dt.strftime("%Y-%m"),
                      "forecast_month": nxt.dt.strftime("%Y-%m")})
    f["pred_bod_mgl"] = (10 ** B["regressor"].predict(X[B["feats"]])).round(2)
    f["prob_polluted"] = B["classifier"].predict_proba(X[B["cfeats"]])[:, 1].round(3)
    f["pred_polluted"] = (f["prob_polluted"] >= 0.5).astype(int)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_dir", default="out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    B = joblib.load(a.model)
    t = B["trained_on"]
    print(f"model trained on {t['n_rows']} rows, {t['n_lakes']} lakes, {t['first_month']} to {t['last_month']}")
    d = pd.read_csv(a.inp, parse_dates=["date"])
    print(f"scoring {len(d)} rows, {d.lake_id.nunique()} lakes from {a.inp}")

    out = baseline_predictions(d, B)
    out.to_csv(os.path.join(a.out_dir, "predictions.csv"), index=False)
    print(f"wrote predictions.csv ({int((out.seen_in_training == 0).sum())} rows from lakes not seen in training)")
    report_accuracy(out, B)

    f = next_month_forecast(d, B)
    if f is not None:
        f.to_csv(os.path.join(a.out_dir, "next_month_forecast.csv"), index=False)
        print(f"\nwrote next_month_forecast.csv ({len(f)} lakes, forecast for {f.forecast_month.iloc[0]})")


if __name__ == "__main__":
    main()
