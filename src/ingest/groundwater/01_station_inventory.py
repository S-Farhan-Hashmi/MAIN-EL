"""Step 1: clip India-WRIS manual groundwater readings to the study belt."""
import pandas as pd

RAW = "groundwater-level-collected-manually.csv"
OUT = "data/interim/groundwater/belt_readings.csv"

# Rough box around Kengeri -> Byramangala -> Bidadi. Verify on a map and adjust.
LON_MIN, LON_MAX = 77.30, 77.55
LAT_MIN, LAT_MAX = 12.70, 12.95

keep = []
for chunk in pd.read_csv(RAW, chunksize=500_000, low_memory=False):
    chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
    sel = chunk[
        chunk.longitude.between(LON_MIN, LON_MAX)
        & chunk.latitude.between(LAT_MIN, LAT_MAX)
    ]
    if len(sel):
        keep.append(sel)

if not keep:
    raise ValueError("No groundwater readings found inside the specified coordinate bounds.")

df = pd.concat(keep, ignore_index=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df.to_csv(OUT, index=False)

print("rows in belt:", len(df))
print("stations:", df.station_code.nunique())
print("\nWhat is measured:")
print(df.data_type_description.value_counts())
print("\nPer station:")
summary = (
    df.groupby(["station_code", "station_name"])
    .agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        well_type=("well_type", "first"),
        agency=("agency_name", "first"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_readings=("data_value", "count"),
    )
    .sort_values("n_readings", ascending=False)
)
print(summary.to_string())
summary.to_csv("data/interim/groundwater/belt_station_summary.csv")
