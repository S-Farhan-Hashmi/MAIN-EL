from pathlib import Path
import gzip
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import rasterio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path.cwd()
STATIONS_FILE = ROOT / "data/processed/groundwater/gw_tidy.csv"
OUT_FILE = ROOT / "data/processed/groundwater/chirps_monthly_2010_2024.csv"
BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_monthly/tifs"
)

START_YEAR, END_YEAR = 2010, 2024
SOURCE = "CHIRPS v2.0 global monthly GeoTIFF"

df = pd.read_csv(STATIONS_FILE)
required = {"station_id", "station_name", "lat", "lon"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing station columns: {sorted(missing)}")

stations = (
    df[["station_id", "station_name", "lat", "lon"]]
    .drop_duplicates("station_id")
    .reset_index(drop=True)
)
if stations.empty:
    raise ValueError("No stations found.")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load existing results; never discard previously collected months.
if OUT_FILE.exists():
    old = pd.read_csv(OUT_FILE)
    print(f"Existing rows loaded: {len(old)}")
else:
    old = pd.DataFrame()

key = ["station_id", "year", "month"]
completed = set()
if not old.empty and all(c in old.columns for c in key):
    completed = set(
        old[key].drop_duplicates().itertuples(index=False, name=None)
    )

# A month is complete only if every expected station has a row.
expected_ids = set(stations["station_id"].astype(str))
completed_months = set()
if not old.empty and all(c in old.columns for c in key):
    tmp = old.copy()
    tmp["station_id"] = tmp["station_id"].astype(str)
    for (year, month), group in tmp.groupby(["year", "month"]):
        if expected_ids.issubset(set(group["station_id"])):
            completed_months.add((int(year), int(month)))

print("Stations:", len(stations))
print(f"Target period: {START_YEAR}-01 to {END_YEAR}-12")
print("Already complete months:", len(completed_months))

retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))

failures = []

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        if (year, month) in completed_months:
            continue

        name = f"chirps-v2.0.{year}.{month:02d}.tif"
        url = f"{BASE_URL}/{name}.gz"
        print(f"Processing {year}-{month:02d}...", flush=True)

        new_rows = []
        try:
            response = session.get(url, timeout=(20, 90))
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                gz_path = tmp / f"{name}.gz"
                tif_path = tmp / name
                gz_path.write_bytes(response.content)

                with gzip.open(gz_path, "rb") as src, open(tif_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                with rasterio.open(tif_path) as raster:
                    coords = list(zip(stations["lon"], stations["lat"]))
                    samples = list(raster.sample(coords, masked=True))

                    for (_, station), sample in zip(stations.iterrows(), samples):
                        raw = sample[0]
                        invalid = (
                            np.ma.is_masked(raw)
                            or not np.isfinite(float(raw))
                            or float(raw) < 0
                        )
                        value = np.nan if invalid else float(raw)

                        new_rows.append({
                            "station_id": station["station_id"],
                            "station_name": station["station_name"],
                            "lat": station["lat"],
                            "lon": station["lon"],
                            "year": year,
                            "month": month,
                            "rainfall_mm": value,
                            "source": SOURCE,
                            "is_imputed": False,
                            "is_valid": not invalid,
                        })

        except Exception as exc:
            failures.append({"year": year, "month": month, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
            # Do not mark failed months complete; a later run can retry them.
            continue

        # Replace only this month's station records, then save all progress.
        if not old.empty:
            old = old[~((old["year"] == year) & (old["month"] == month))]
        old = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
        old.to_csv(OUT_FILE, index=False)
        completed_months.add((year, month))
        print(f"  Saved {len(new_rows)} station values.", flush=True)
        time.sleep(0.2)

session.close()

print("\n=== RESUME SUMMARY ===")
print("Output:", OUT_FILE)
print("Total rows:", len(old))
if not old.empty:
    print("Years present:", sorted(old["year"].dropna().unique().tolist()))
    print("Months present per year:")
    print(old.groupby("year")["month"].nunique().to_string())
print("Complete months:", len(completed_months))
print("Failed months this run:", len(failures))

if failures:
    failure_file = OUT_FILE.with_name("chirps_failed_months.csv")
    pd.DataFrame(failures).to_csv(failure_file, index=False)
    print("Failure log:", failure_file)
