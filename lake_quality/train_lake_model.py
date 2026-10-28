#!/usr/bin/env python3
"""
Phase 5 - lake quality model (honest comparison against baselines).

Target: BOD (reported as log10 BOD; 'polluted' = BOD >= 10 mg/L).
Always trains the simple baselines first, then ML, so you can show whether ML helps.

  python train_lake_model.py --train out/lake_clean_train.csv --test out/lake_clean_test.csv --out_dir out

Four experiments
  1. Next-month forecast, time split   (features: last month's chemistry + season + population)
  2. Next-month 'polluted' state       (same features, classification)
  3. Unseen-lake check                 (features: population + location + season only, grouped by lake)
  4. Final test on July 2026           (no last-month data exists, so lake history + drivers only)

Saves out/lake_model.joblib, which predict_lake.py loads to score new data.

Leakage rules followed: the KSPCB use class is never used (it is derived from that month's DO/BOD),
and no same-month chemistry is used to predict that month's BOD.
"""
import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

POOR_BOD = 10.0
Y = "log_bod_mgl"
# only parameters that are fully measured in every month (no imputed Sept/May columns)
LAG_COLS = ["log_bod_mgl", "log_cod_mgl", "log_fecal_coliform", "log_total_coliform", "ph", "log_nitrate_n"]
DRIVERS = ["log_pop_1km", "log_pop_2km", "lat", "lon", "month_sin", "month_cos"]
RESULTS = []


def reg_row(task, model, y, p, n_note=""):
    RESULTS.append(dict(task=task, model=model, n=len(y), rmse_log10=np.sqrt(mean_squared_error(y, p)),
                        mae_log10=mean_absolute_error(y, p), r2=r2_score(y, p),
                        mae_mgl=mean_absolute_error(10 ** y, 10 ** p), note=n_note))


def clf_row(task, model, y, p, score=None, n_note=""):
    RESULTS.append(dict(task=task, model=model, n=len(y), accuracy=accuracy_score(y, p),
                        f1_polluted=f1_score(y, p, zero_division=0),
                        auc=roc_auc_score(y, score) if score is not None and len(set(y)) > 1 else np.nan,
                        note=n_note))


def make_pairs(d):
    """One row per lake and month with the PREVIOUS month's chemistry attached."""
    d = d.sort_values(["lake_id", "date"]).copy()
    d["poor"] = (10 ** d[Y] >= POOR_BOD).astype(int)
    for c in LAG_COLS + ["poor"]:
        d["lag_" + c] = d.groupby("lake_id")[c].shift(1)
    prev = d.groupby("lake_id")["date"].shift(1)
    d["gap"] = (d["date"].dt.year - prev.dt.year) * 12 + (d["date"].dt.month - prev.dt.month)
    pairs = d[d["gap"] == 1].dropna(subset=["lag_" + c for c in LAG_COLS]).copy()
    return d, pairs


def forecast_experiments(train, cut):
    d, pairs = make_pairs(train)
    tr, te = pairs[pairs.date < cut], pairs[pairs.date >= cut]
    print(f"[1/2] next-month forecast: train {len(tr)} lake-months, test {len(te)} (from {cut.date()})")
    feats = ["lag_" + c for c in LAG_COLS] + ["log_pop_1km", "log_pop_2km", "month_sin", "month_cos", "lat", "lon"]
    t1 = "forecast next-month log BOD"
    reg_row(t1, "baseline: global mean", te[Y], np.full(len(te), tr[Y].mean()))
    reg_row(t1, "baseline: persistence (last month)", te[Y], te["lag_log_bod_mgl"])
    lake_mean = tr.groupby("lake_id")[Y].mean()
    reg_row(t1, "baseline: lake mean (train)", te[Y], te.lake_id.map(lake_mean).fillna(tr[Y].mean()))
    for name, m in {"ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
                    "gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                                   subsample=0.8, random_state=0)}.items():
        m.fit(tr[feats], tr[Y])
        reg_row(t1, "ML: " + name, te[Y], m.predict(te[feats]))

    t2 = "forecast next-month polluted (BOD>=10)"
    reg_state = tr.groupby("lake_id").poor.mean()
    s = te.lake_id.map(reg_state).fillna(tr.poor.mean())
    clf_row(t2, "baseline: always 'not polluted'", te.poor, np.zeros(len(te), int))
    clf_row(t2, "baseline: persistence (last month)", te.poor, te.lag_poor.astype(int), te.lag_poor)
    clf_row(t2, "baseline: lake share >= 0.5 (train)", te.poor, (s >= 0.5).astype(int), s)
    cfeats = feats + ["lag_poor"]
    for name, m in {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
                    "gradient boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                                    random_state=0)}.items():
        m.fit(tr[cfeats], tr.poor)
        clf_row(t2, "ML: " + name, te.poor, m.predict(te[cfeats]), m.predict_proba(te[cfeats])[:, 1])


def unseen_lake_experiment(train):
    d = train.dropna(subset=[Y]).reset_index(drop=True)
    print(f"[3] unseen-lake check: {d.lake_id.nunique()} lakes, 5-fold grouped by lake")
    task = "unseen lakes: log BOD from population+location+season"
    gkf = GroupKFold(n_splits=5)
    base, ridge, gbr = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for tr, te in gkf.split(d, groups=d.lake_id):
        base[te] = d.iloc[tr][Y].mean()
        ridge[te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(d.iloc[tr][DRIVERS], d.iloc[tr][Y]).predict(d.iloc[te][DRIVERS])
        gbr[te] = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                            random_state=0).fit(d.iloc[tr][DRIVERS], d.iloc[tr][Y]).predict(d.iloc[te][DRIVERS])
    reg_row(task, "baseline: global mean", d[Y], base)
    reg_row(task, "ML: ridge", d[Y], ridge)
    reg_row(task, "ML: gradient boosting", d[Y], gbr)
    lm = d.groupby("lake_id")[Y].mean()
    print("      correlation of lake-mean log BOD with log_pop_1km: %.2f, with log_pop_2km: %.2f" %
          (pd.concat([lm, d.groupby("lake_id").log_pop_1km.first()], axis=1).corr().iloc[0, 1],
           pd.concat([lm, d.groupby("lake_id").log_pop_2km.first()], axis=1).corr().iloc[0, 1]))


def final_test(train, test):
    print(f"[4] final test: {len(test)} lakes")
    t = test.dropna(subset=[Y]).copy()
    seen = t.lake_id.isin(train.lake_id)
    task = "FINAL TEST July 2026: log BOD"
    lm = train.groupby("lake_id")[Y].mean()
    reg_row(task, "baseline: global mean (train)", t[Y], np.full(len(t), train[Y].mean()))
    p = t.lake_id.map(lm).fillna(train[Y].mean())
    reg_row(task, "baseline: lake mean (train)", t[Y], p, f"{int(seen.sum())} of {len(t)} lakes seen in training")
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    m.fit(train[DRIVERS], train[Y])
    reg_row(task, "ML: gradient boosting (drivers only)", t[Y], m.predict(t[DRIVERS]),
            "lat/lon act as a lake ID here, so this is not evidence that population helps")

    t["poor"] = (10 ** t[Y] >= POOR_BOD).astype(int)
    tr = train.copy()
    tr["poor"] = (10 ** tr[Y] >= POOR_BOD).astype(int)
    share = tr.groupby("lake_id").poor.mean()
    s = t.lake_id.map(share).fillna(tr.poor.mean())
    task = "FINAL TEST July 2026: polluted (BOD>=10)"
    clf_row(task, "baseline: always 'not polluted'", t.poor, np.zeros(len(t), int))
    clf_row(task, "baseline: lake share >= 0.5 (train)", t.poor, (s >= 0.5).astype(int), s)


def write_baseline_table(train, out_dir):
    tr = train.copy()
    tr["poor"] = (10 ** tr[Y] >= POOR_BOD).astype(int)
    tab = tr.groupby("lake_id").agg(n_months=(Y, "size"), mean_log10_bod=(Y, "mean"), std_log10_bod=(Y, "std"),
                                    polluted_share=("poor", "mean"), lat=("lat", "first"), lon=("lon", "first"),
                                    log_pop_1km=("log_pop_1km", "first"), log_pop_2km=("log_pop_2km", "first")).reset_index()
    tab["typical_bod_mgl"] = 10 ** tab["mean_log10_bod"]
    tab = tab.round(4)
    tab.to_csv(os.path.join(out_dir, "lake_baseline_table.csv"), index=False)
    print(f"      wrote lake_baseline_table.csv ({len(tab)} lakes)")
    return tab


def save_model(train, tab, out_dir):
    """Fit the final models on ALL training months and save everything predict_lake.py needs."""
    d, pairs = make_pairs(train)
    feats = ["lag_" + c for c in LAG_COLS] + ["log_pop_1km", "log_pop_2km", "month_sin", "month_cos", "lat", "lon"]
    cfeats = feats + ["lag_poor"]
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    random_state=0).fit(pairs[feats], pairs[Y])
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     random_state=0).fit(pairs[cfeats], pairs.poor)
    bundle = dict(baseline=tab, global_mean_log_bod=float(train[Y].mean()),
                  global_polluted_share=float((10 ** train[Y] >= POOR_BOD).mean()),
                  regressor=reg, classifier=clf, feats=feats, cfeats=cfeats, lag_cols=LAG_COLS,
                  poor_bod=POOR_BOD,
                  trained_on=dict(first_month=str(train.date.min().date()), last_month=str(train.date.max().date()),
                                  n_lakes=int(train.lake_id.nunique()), n_rows=int(len(train)),
                                  n_forecast_pairs=int(len(pairs))))
    path = os.path.join(out_dir, "lake_model.joblib")
    joblib.dump(bundle, path)
    print(f"      wrote lake_model.joblib (baseline table + next-month forecast models, trained on {len(pairs)} pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--cut", default="2025-09-01", help="first month of the time-split test period")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    train = pd.read_csv(a.train, parse_dates=["date"])
    test = pd.read_csv(a.test, parse_dates=["date"])
    forecast_experiments(train, pd.Timestamp(a.cut))
    unseen_lake_experiment(train)
    final_test(train, test)
    tab = write_baseline_table(train, a.out_dir)
    save_model(train, tab, a.out_dir)

    res = pd.DataFrame(RESULTS)
    res.round(3).to_csv(os.path.join(a.out_dir, "model_results.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 48)
    for task, g in res.groupby("task", sort=False):
        print("\n" + task)
        cols = [c for c in ["model", "n", "rmse_log10", "mae_log10", "r2", "mae_mgl", "accuracy", "f1_polluted", "auc", "note"]
                if g[c].notna().any()]
        print(g[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
