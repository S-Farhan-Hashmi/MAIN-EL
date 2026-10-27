"""Step 3a: monthly CHIRPS rainfall at each well via NOAA ERDDAP."""
import io, requests, pandas as pd

BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalMonthlyP05"
wells = (pd.read_csv("data/processed/groundwater/gw_seasonal.csv")
         [["station_id", "station_name", "lat", "lon"]].drop_duplicates())

# 1) check the metadata first
r = requests.get(BASE + ".das", timeout=60)
if not r.ok:
    print(r.status_code, r.text[:500]); raise SystemExit("could not read metadata")
txt = r.text
i = txt.find("precip {")
if i < 0:
    print(txt[:3000]); raise SystemExit("no 'precip' variable: paste this output to me")
block = txt[i:i + 600]
print(block)
for key in ("time_coverage_start", "time_coverage_end"):
    j = txt.find(key); print(txt[j:j + 80])
per_day = "mm/day" in block
print("\nunits are per day:", per_day)

# 2) one request per well (nearest grid cell)
frames = []
for w in wells.itertuples():
    q = f".csv?precip[(1985-01-01):1:(2024-12-31)][({w.lat})][({w.lon})]"
    resp = requests.get(BASE + q, timeout=120)
    if not resp.ok:
        print(w.station_name, resp.status_code, resp.text[:500]); raise SystemExit
    d = pd.read_csv(io.StringIO(resp.text), skiprows=[1])  # row 2 is the units line
    d["station_id"] = w.station_id
    d["station_name"] = w.station_name
    frames.append(d)

rain = pd.concat(frames, ignore_index=True)
rain["time"] = pd.to_datetime(rain["time"]).dt.tz_localize(None)
rain["days"] = rain["time"].dt.days_in_month
rain["rain_mm"] = rain["precip"] * rain["days"] if per_day else rain["precip"]
print("\nmissing values:", rain.rain_mm.isna().sum())
rain.to_csv("data/raw/rainfall/chirps_monthly_wells.csv", index=False)

# 3) sanity check: annual totals should be roughly 800 to 1000 mm
rain["year"] = rain.time.dt.year
ann = rain[rain.year <= 2023].groupby(["station_name", "year"]).rain_mm.sum()
print("\nmean annual rainfall (mm) per well, 1985 to 2023:")
print(ann.groupby("station_name").mean().round(0).to_string())
print("\ngrid cells used:")
print(rain.groupby("station_name")[["latitude", "longitude"]].first().to_string())
