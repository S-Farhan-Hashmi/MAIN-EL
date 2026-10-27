"""Step 2: Clean groundwater readings, inspect gaps, and plot stations."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "data/interim/groundwater/belt_readings.csv"
OUT = "data/processed/groundwater/gw_tidy.csv"
FIG_DIR = "notebooks/groundwater/figs"

df = pd.read_csv(RAW, low_memory=False, parse_dates=["date"])

df = df[df["data_type_description"] == "MANUAL-Water Level"].copy()
df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

print("Raw manual-water-level rows:", len(df))
print("Missing data values:", df["data_value"].isna().sum())
print("Missing dates:", df["date"].isna().sum())
print("Duplicate station-date rows:", df.duplicated(
    ["station_code", "date"], keep=False
).sum())
print("Numeric value range:", df["data_value"].min(), "to",
      df["data_value"].max())

# Remove rows without a usable date or measurement.
df = df.dropna(subset=["data_value", "date", "station_code"])

# Keep one row per station and date.
# If duplicates have conflicting measurements, inspect them separately.
df = df.drop_duplicates(["station_code", "date"], keep="first").copy()

tidy = pd.DataFrame({
    "station_id": df["station_code"].astype(str),
    "station_name": df["station_name"],
    "lat": df["latitude"],
    "lon": df["longitude"],
    "well_type": df["well_type"],
    "date": df["date"],
    "depth_to_water_m": df["data_value"],
    "source": "India-WRIS/CGWB; provenance to be verified",
    "is_imputed": False,
}).sort_values(["station_id", "date"])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

tidy.to_csv(OUT, index=False)
print("\nCleaned rows:", len(tidy))
print("Stations:", tidy["station_id"].nunique())
print("Saved:", OUT)

# Count readings per station per year.
tidy["year"] = tidy["date"].dt.year
year_table = tidy.pivot_table(
    index="station_name",
    columns="year",
    values="depth_to_water_m",
    aggfunc="count",
    fill_value=0,
).astype(int)

print("\nReadings per station per year:")
print(year_table.to_string())
year_table.to_csv("data/processed/groundwater/readings_per_year.csv")

# Plot individual station time series.
for station_id, g in tidy.groupby("station_id"):
    g = g.sort_values("date")
    name = str(g["station_name"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(g["date"], g["depth_to_water_m"], marker="o", ms=3, lw=1)
    ax.set_title(f"{name} ({g['well_type'].iloc[0]})")
    ax.set_ylabel("Reported water level (unit to verify)")
    ax.set_xlabel("Date")

    # Do not assume whether increasing values mean deeper water
    # until the source metadata and units are verified.
    fig.tight_layout()
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in name
    )
    fig.savefig(f"{FIG_DIR}/{station_id}_{safe_name}.png", dpi=110)
    plt.close(fig)

print("\nPlots saved in:", FIG_DIR)
