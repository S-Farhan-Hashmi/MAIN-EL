"""
build_features.py
-----------------
Feature engineering pipeline for the pilot corridor soil-moisture / NDVI dataset.

Steps
-----
1.  Load raw CSVs + watershed GeoJSON
2.  Temporal alignment  - expand every source to a full daily grid per watershed;
                          slow sources (S2, MODIS) are forward-filled and flagged
3.  QC                  - range checks, duplicate detection, unit standardisation
4.  Gap-fill NDVI       - use MODIS to fill S2 gaps; flag filled rows
5.  Missing-data flags  - explicit boolean columns for every imputed value
6.  Feature engineering - rolling stats, lag features, seasonal flags,
                          static proxy features (borewell_density etc.)
7.  Write master feature table  -> data/processed/master_features.csv

Usage
-----
  python src/features/build_features.py

  Optional env vars:
    RAW_DIR       path to raw CSVs          (default: data/raw)
    PROCESSED_DIR path for output           (default: data/processed)
    GIS_DIR       path to watershed GeoJSON (default: gis)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR       = Path(os.environ.get("RAW_DIR",       "data/raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", "data/processed"))
GIS_DIR       = Path(os.environ.get("GIS_DIR",       "gis"))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Valid physical ranges  (inclusive)
RANGE_CHECKS = {
    "NDVI_S2":    (-1.0,  1.0),
    "NDVI_MODIS": (-1.0,  1.0),
    "SM_ERA5":    ( 0.0,  1.0),
}

# Static proxy features per locality
# (replace / extend with real census / survey data when available)
PROXY_TABLE = pd.DataFrame([
    {
        "watershed_id":        "LOC_01",
        "watershed_name":      "Bang_Uni_Ars_Ls",
        "area_km2":            1.8,
        "borewell_density":    4.2,    # borewells per km2 (survey estimate)
        "irrigated_area_pct":  12.5,   # % of agricultural / green area irrigated
        "population_density":  1850.0, # persons per km2
        "dominant_crop":       "lawn_ornamental",
        "soil_type":           "red_laterite",
    },
    {
        "watershed_id":        "LOC_02",
        "watershed_name":      "Bang_Uni_Ars_Us",
        "area_km2":            1.6,
        "borewell_density":    3.8,
        "irrigated_area_pct":  10.0,
        "population_density":  1620.0,
        "dominant_crop":       "lawn_ornamental",
        "soil_type":           "red_laterite",
    },
    {
        "watershed_id":        "LOC_03",
        "watershed_name":      "Suggaunahalli",
        "area_km2":            3.2,
        "borewell_density":    8.6,
        "irrigated_area_pct":  28.4,
        "population_density":  740.0,
        "dominant_crop":       "vegetables_horticulture",
        "soil_type":           "red_laterite",
    },
    {
        "watershed_id":        "LOC_04",
        "watershed_name":      "Bidadi",
        "area_km2":            18.5,
        "borewell_density":    14.2,
        "irrigated_area_pct":  45.8,
        "population_density":  320.0,
        "dominant_crop":       "paddy_ragi",
        "soil_type":           "red_sandy_loam",
    },
    {
        "watershed_id":        "LOC_05",
        "watershed_name":      "Hejjala",
        "area_km2":            8.4,
        "borewell_density":    11.9,
        "irrigated_area_pct":  40.1,
        "population_density":  280.0,
        "dominant_crop":       "paddy_sugarcane",
        "soil_type":           "red_sandy_loam",
    },
    {
        "watershed_id":        "LOC_06",
        "watershed_name":      "Thalaghattapura",
        "area_km2":            6.7,
        "borewell_density":    7.3,
        "irrigated_area_pct":  22.6,
        "population_density":  560.0,
        "dominant_crop":       "vegetables_horticulture",
        "soil_type":           "black_cotton",
    },
    {
        "watershed_id":        "LOC_07",
        "watershed_name":      "Kengeri",
        "area_km2":            14.2,
        "borewell_density":    5.1,
        "irrigated_area_pct":  15.3,
        "population_density":  2140.0,
        "dominant_crop":       "vegetables_horticulture",
        "soil_type":           "black_cotton",
    },
    {
        "watershed_id":        "LOC_08",
        "watershed_name":      "Jnanabharathi",
        "area_km2":            4.9,
        "borewell_density":    4.7,
        "irrigated_area_pct":  18.0,
        "population_density":  1200.0,
        "dominant_crop":       "lawn_ornamental",
        "soil_type":           "red_laterite",
    },
])

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s2    = pd.read_csv(RAW_DIR / "ndvi_sentinel2.csv",   parse_dates=["date"])
    modis = pd.read_csv(RAW_DIR / "ndvi_modis.csv",       parse_dates=["date"])
    era5  = pd.read_csv(RAW_DIR / "soilmoisture_era5.csv", parse_dates=["date"])
    return s2, modis, era5


# ---------------------------------------------------------------------------
# 2. Temporal alignment - build a full daily grid per watershed
# ---------------------------------------------------------------------------
def make_daily_grid(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Find the global date range across all sources and all watersheds,
    then return a DataFrame with one row per (watershed_id, date).
    """
    dates      = pd.concat([d["date"] for d in all_dfs])
    watersheds = pd.concat([d["watershed_id"] for d in all_dfs]).unique()
    date_range = pd.date_range(dates.min(), dates.max(), freq="D")

    grid = pd.MultiIndex.from_product(
        [watersheds, date_range], names=["watershed_id", "date"]
    )
    return pd.DataFrame(index=grid).reset_index()


def align_to_grid(
    grid: pd.DataFrame,
    df: pd.DataFrame,
    value_col: str,
    ffill_limit: int = 16,
) -> tuple[pd.DataFrame, str]:
    """
    Left-join *df* onto *grid*, forward-fill up to *ffill_limit* days,
    and add a boolean flag column indicating which rows were forward-filled.
    Returns (merged_df, flag_col_name).
    """
    flag_col = f"{value_col}_ffilled"
    merged = grid.merge(
        df[["watershed_id", "date", value_col]],
        on=["watershed_id", "date"],
        how="left",
    )
    # Mark rows that originally had no value
    merged[flag_col] = merged[value_col].isna()

    # Forward-fill within each watershed, respecting limit
    merged = merged.sort_values(["watershed_id", "date"])
    merged[value_col] = (
        merged.groupby("watershed_id")[value_col]
        .transform(lambda s: s.ffill(limit=ffill_limit))
    )
    return merged, flag_col


# ---------------------------------------------------------------------------
# 3. QC helpers
# ---------------------------------------------------------------------------
def qc_range(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    """Return boolean mask of out-of-range values."""
    mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
    n = mask.sum()
    if n:
        print(f"  [QC] {col}: {n} values outside [{lo}, {hi}] — clipped")
    return mask


def remove_duplicates(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [QC] Dropped {n_dropped} duplicate rows on {subset}")
    return df


# ---------------------------------------------------------------------------
# 4. Gap-fill NDVI: use MODIS where S2 is missing
# ---------------------------------------------------------------------------
def gap_fill_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine S2 and MODIS NDVI into a single NDVI column.
    S2 is preferred; MODIS fills gaps.  A flag column records the source.
    """
    conditions = [
        df["NDVI_S2"].notna(),
        df["NDVI_MODIS"].notna(),
    ]
    choices = [
        df["NDVI_S2"],
        df["NDVI_MODIS"],
    ]
    df["NDVI"] = np.select(conditions, choices, default=np.nan)
    df["NDVI_source"] = np.select(
        conditions,
        ["Sentinel-2", "MODIS_gap_fill"],
        default="missing",
    )
    df["NDVI_gap_filled"] = df["NDVI_source"] == "MODIS_gap_fill"
    df["NDVI_missing"]    = df["NDVI_source"] == "missing"
    return df


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame, col: str, lags: list[int] = [1, 3, 7]
) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}d"] = (
            df.groupby("watershed_id")[col].shift(lag)
        )
    return df


def add_rolling_features(
    df: pd.DataFrame, col: str, windows: list[int] = [3, 7, 14]
) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll{w}d_mean"] = (
            df.groupby("watershed_id")[col]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"{col}_roll{w}d_std"] = (
            df.groupby("watershed_id")[col]
            .transform(lambda s: s.rolling(w, min_periods=2).std())
        )
    return df


def add_seasonal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Karnataka monsoon calendar flags (SW monsoon).
    Adjust onset_doy / withdrawal_doy for other years as needed.
    """
    doy = df["date"].dt.dayofyear
    month = df["date"].dt.month

    # SW monsoon arrives Bengaluru ~June 5 (doy 157), withdraws ~Oct 20 (doy 293)
    MONSOON_ONSET_DOY      = 157
    MONSOON_WITHDRAWAL_DOY = 293

    df["is_pre_monsoon"]  = (doy < MONSOON_ONSET_DOY) & (month.between(3, 5))
    df["is_monsoon"]      = doy.between(MONSOON_ONSET_DOY, MONSOON_WITHDRAWAL_DOY)
    df["is_post_monsoon"] = (doy > MONSOON_WITHDRAWAL_DOY) & (month.between(10, 11))
    df["is_dry_season"]   = month.isin([12, 1, 2])

    # Approximate monsoon week (1 = first week of monsoon, etc.)
    df["monsoon_week"] = np.where(
        df["is_monsoon"],
        ((doy - MONSOON_ONSET_DOY) // 7 + 1).clip(lower=1),
        np.nan,
    )
    return df


def add_sm_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple water-stress index: how far SM is below the monthly maximum.
    Higher value = more stressed. Useful as a proxy irrigation trigger.
    """
    monthly_max = (
        df.groupby(["watershed_id", df["date"].dt.to_period("M")])["SM_ERA5"]
        .transform("max")
    )
    df["SM_stress_index"] = (monthly_max - df["SM_ERA5"]) / monthly_max.replace(0, np.nan)
    return df


def add_ndvi_change(df: pd.DataFrame) -> pd.DataFrame:
    """Day-on-day NDVI change rate (greening / browning signal)."""
    df["NDVI_delta1d"] = df.groupby("watershed_id")["NDVI"].diff(1)
    df["NDVI_delta7d"] = df.groupby("watershed_id")["NDVI"].diff(7)
    return df


# ---------------------------------------------------------------------------
# 6. Master pipeline
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 60)
    print("Feature pipeline  -  pilot corridor")
    print("=" * 60)

    # ---- Load --------------------------------------------------------------
    print("\n[1/7] Loading raw data ...")
    s2, modis, era5 = load_raw()
    print(f"  S2:    {len(s2)} rows | MODIS: {len(modis)} rows | ERA5: {len(era5)} rows")

    # ---- QC on raw ---------------------------------------------------------
    print("\n[2/7] Quality control ...")
    s2    = remove_duplicates(s2,    ["watershed_id", "date"])
    modis = remove_duplicates(modis, ["watershed_id", "date"])
    era5  = remove_duplicates(era5,  ["watershed_id", "date"])

    for col, (lo, hi) in RANGE_CHECKS.items():
        src = s2 if "S2" in col else (modis if "MODIS" in col else era5)
        if col in src.columns:
            bad = qc_range(src, col, lo, hi)
            src.loc[bad, col] = src.loc[bad, col].clip(lo, hi)

    # ---- Daily grid --------------------------------------------------------
    print("\n[3/7] Building daily grid ...")
    grid = make_daily_grid([s2, modis, era5])
    print(f"  Grid: {len(grid)} rows  ({grid.watershed_id.nunique()} watersheds "
          f"× {grid.date.nunique()} days)")

    # ---- Temporal alignment + forward-fill ---------------------------------
    print("\n[4/7] Aligning sources to daily grid (forward-fill slow sources) ...")

    # S2: forward-fill up to 5 days (revisit cadence)
    s2_grid, s2_flag    = align_to_grid(grid, s2,    "NDVI_S2",    ffill_limit=5)
    # MODIS: forward-fill up to 16 days (composite period)
    modis_grid, m_flag  = align_to_grid(grid, modis, "NDVI_MODIS", ffill_limit=16)
    # ERA5: daily - forward-fill only 1 day for any end-of-month gaps
    era5_grid, e_flag   = align_to_grid(grid, era5,  "SM_ERA5",    ffill_limit=1)

    print(f"  NDVI_S2    forward-filled on {s2_grid[s2_flag].sum()} / {len(s2_grid)} rows")
    print(f"  NDVI_MODIS forward-filled on {modis_grid[m_flag].sum()} / {len(modis_grid)} rows")
    print(f"  SM_ERA5    forward-filled on {era5_grid[e_flag].sum()} / {len(era5_grid)} rows")

    # ---- Merge all sources -------------------------------------------------
    df = grid.copy()
    df = df.merge(s2_grid[   ["watershed_id","date","NDVI_S2",    s2_flag]], on=["watershed_id","date"], how="left")
    df = df.merge(modis_grid[["watershed_id","date","NDVI_MODIS", m_flag]],  on=["watershed_id","date"], how="left")
    df = df.merge(era5_grid[ ["watershed_id","date","SM_ERA5",    e_flag]],  on=["watershed_id","date"], how="left")

    # ---- Gap-fill NDVI (S2 preferred, MODIS fallback) ----------------------
    print("\n[5/7] Gap-filling NDVI (S2 -> MODIS fallback) ...")
    df = gap_fill_ndvi(df)
    print(f"  NDVI source breakdown:\n{df.NDVI_source.value_counts().to_string()}")

    # Explicit missing-value flags for target column (SM_ERA5)
    df["SM_ERA5_missing"] = df["SM_ERA5"].isna()

    # ---- Feature engineering -----------------------------------------------
    print("\n[6/7] Engineering features ...")

    # Seasonal flags
    df = add_seasonal_flags(df)

    # SM features
    df = add_lag_features(df,    "SM_ERA5",  lags=[1, 3, 7])
    df = add_rolling_features(df,"SM_ERA5",  windows=[3, 7, 14])
    df = add_sm_stress_index(df)

    # NDVI features
    df = add_lag_features(df,    "NDVI",     lags=[1, 3, 7])
    df = add_rolling_features(df,"NDVI",     windows=[3, 7])
    df = add_ndvi_change(df)

    # Time index features
    df["doy"]        = df["date"].dt.dayofyear
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year

    # Cyclical encoding of day-of-year (avoids discontinuity at year boundary)
    df["doy_sin"]    = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"]    = np.cos(2 * np.pi * df["doy"] / 365)

    # ---- Proxy / static features -------------------------------------------
    df = df.merge(PROXY_TABLE, on="watershed_id", how="left")

    # ---- Final column ordering ---------------------------------------------
    id_cols    = ["watershed_id", "watershed_name", "date", "year", "month",
                  "week", "doy", "doy_sin", "doy_cos"]
    flag_cols  = [c for c in df.columns if c.endswith(("_ffilled","_missing",
                                                        "_gap_filled"))]
    source_cols= ["NDVI_source"]
    raw_cols   = ["SM_ERA5", "NDVI_S2", "NDVI_MODIS", "NDVI"]
    feat_cols  = [c for c in df.columns if c not in
                  id_cols + flag_cols + source_cols + raw_cols
                  + ["data_source", "note", "gap_filled"]
                  and not c.startswith("is_")]
    season_cols= [c for c in df.columns if c.startswith("is_") or c == "monsoon_week"]
    proxy_cols = ["area_km2","borewell_density","irrigated_area_pct",
                  "population_density","dominant_crop","soil_type"]

    ordered = id_cols + raw_cols + feat_cols + season_cols + proxy_cols + source_cols + flag_cols
    # keep only columns that actually exist
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]

    # ---- Write output ------------------------------------------------------
    print("\n[7/7] Writing master feature table ...")
    out_path = PROCESSED_DIR / "master_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> {out_path}")
    print(f"     {len(df)} rows  ×  {len(df.columns)} columns")

    # ---- Quick summary ------------------------------------------------------
    print("\n--- Column inventory ---")
    col_groups = {
        "Identifiers":        id_cols,
        "Raw signals":        [c for c in raw_cols if c in df.columns],
        "SM features":        [c for c in df.columns if "SM_ERA5" in c and c not in raw_cols+flag_cols],
        "NDVI features":      [c for c in df.columns if "NDVI" in c and c not in raw_cols+flag_cols+source_cols],
        "Seasonal flags":     season_cols,
        "Proxy (static)":     [c for c in proxy_cols if c in df.columns],
        "Data-quality flags": flag_cols + [source_cols[0]],
    }
    for group, cols in col_groups.items():
        present = [c for c in cols if c in df.columns]
        if present:
            print(f"  {group} ({len(present)}): {', '.join(present)}")

    print(f"\n  Missing-value summary:")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss.to_string() if len(miss) else "    None")

    print("\n[DONE]")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()

