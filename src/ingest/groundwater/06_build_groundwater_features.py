from pathlib import Path

import pandas as pd

ROOT = Path.cwd()

GW_FILE = ROOT / "data/processed/groundwater/gw_tidy.csv"
RAIN_FILE = ROOT / "data/processed/groundwater/chirps_monthly_2010_2024.csv"
OUT_FILE = ROOT / "data/processed/groundwater/gw_rainfall_features.csv"

# Load station IDs as strings to preserve exact identifiers.
gw = pd.read_csv(GW_FILE, dtype={"station_id": str})
rain = pd.read_csv(RAIN_FILE, dtype={"station_id": str})

required_gw = {"station_id", "date", "depth_to_water_m"}
required_rain = {
    "station_id", "year", "month", "rainfall_mm", "is_valid"
}

if not required_gw.issubset(gw.columns):
    raise ValueError(f"Groundwater columns missing: {required_gw - set(gw.columns)}")

if not required_rain.issubset(rain.columns):
    raise ValueError(f"Rainfall columns missing: {required_rain - set(rain.columns)}")

# Validate rainfall before feature engineering.
rain["is_valid"] = (
    rain["is_valid"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)
rain["rainfall_mm"] = pd.to_numeric(rain["rainfall_mm"], errors="coerce")

if rain.duplicated(["station_id", "year", "month"]).any():
    raise ValueError("Duplicate rainfall station-month records found.")

if rain["rainfall_mm"].isna().any() or not rain["is_valid"].all():
    raise ValueError("Rainfall data contains missing or invalid records.")

if (rain["rainfall_mm"] < 0).any():
    raise ValueError("Negative rainfall values found.")

# Parse groundwater observations.
gw["date"] = pd.to_datetime(gw["date"], errors="coerce")
gw["depth_to_water_m"] = pd.to_numeric(
    gw["depth_to_water_m"], errors="coerce"
)

if gw["date"].isna().any():
    raise ValueError("Groundwater file contains unparseable dates.")

if gw["depth_to_water_m"].isna().any():
    raise ValueError("Groundwater file contains missing/non-numeric targets.")

# Match station IDs and create monthly rainfall lookup series.
rain["month_period"] = pd.PeriodIndex(
    pd.to_datetime(
        dict(year=rain["year"], month=rain["month"], day=1)
    ),
    freq="M",
)

rain_lookup = {}
for station_id, group in rain.groupby("station_id"):
    rain_lookup[station_id] = group.set_index("month_period")["rainfall_mm"]

rain_ids = set(rain_lookup)
gw_ids = set(gw["station_id"])
unmatched = sorted(gw_ids - rain_ids)

# Some groundwater stations may not have matching rainfall locations.
# Keep those observations, but their rainfall features will remain missing.
print("Groundwater observations:", len(gw))
print("Groundwater stations:", len(gw_ids))
print("Stations without rainfall series:", unmatched)

def get_complete_sum(series, periods):
    """Return a sum only when every required prior month is available."""
    values = series.reindex(periods)
    if len(values) != len(periods) or values.isna().any():
        return float("nan")
    return float(values.sum())

def season_for_month(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre_monsoon"
    if month in (6, 7, 8, 9):
        return "southwest_monsoon"
    return "post_monsoon"

features = []

for _, obs in gw.iterrows():
    sid = obs["station_id"]
    observation_month = obs["date"].to_period("M")
    series = rain_lookup.get(sid)

    row = {
        "rainfall_lag_1m_mm": float("nan"),
        "rainfall_prev_3m_sum_mm": float("nan"),
        "rainfall_prev_6m_sum_mm": float("nan"),
        "rainfall_prev_12m_sum_mm": float("nan"),
        "rainfall_months_available_prior_12m": 0,
    }

    if series is not None:
        # Strictly exclude the groundwater observation's calendar month.
        prior_12 = pd.period_range(
            end=observation_month - 1, periods=12, freq="M"
        )
        history = series.reindex(prior_12)

        row["rainfall_months_available_prior_12m"] = int(history.notna().sum())

        lag_month = observation_month - 1
        if lag_month in series.index:
            row["rainfall_lag_1m_mm"] = float(series.loc[lag_month])

        row["rainfall_prev_3m_sum_mm"] = get_complete_sum(
            series, prior_12[-3:]
        )
        row["rainfall_prev_6m_sum_mm"] = get_complete_sum(
            series, prior_12[-6:]
        )
        row["rainfall_prev_12m_sum_mm"] = get_complete_sum(
            series, prior_12
        )

    row.update({
        "observation_month": observation_month.month,
        "observation_year": observation_month.year,
        "season": season_for_month(obs["date"].month),
    })
    features.append(row)

feature_df = pd.DataFrame(features, index=gw.index)
result = pd.concat([gw.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT_FILE, index=False)

print("\nFeature table saved:", OUT_FILE)
print("Rows:", len(result))
print("Columns:", len(result.columns))
print("\nMissing values in rainfall features:")
print(result[[
    "rainfall_lag_1m_mm",
    "rainfall_prev_3m_sum_mm",
    "rainfall_prev_6m_sum_mm",
    "rainfall_prev_12m_sum_mm",
]].isna().sum().to_string())
print("\nSeason counts:")
print(result["season"].value_counts().to_string())
print("\nPreview:")
print(result.head(5).to_string(index=False))
