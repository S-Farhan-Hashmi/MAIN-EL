"""Step 2b: check duplicates and 1996, label seasons, build seasonal table."""
import pandas as pd

raw = pd.read_csv("data/interim/groundwater/belt_readings.csv", parse_dates=["date"])
raw = raw[raw.data_type_description == "MANUAL-Water Level"]

# 1) duplicates: do the values agree?
d = raw[raw.duplicated(["station_code", "date"], keep=False)]
print("DUPLICATE ROWS:")
print(d.sort_values(["station_code", "date"])[
    ["station_name", "date", "data_value", "time"]].to_string())

# 2) the 1996 oddity
r96 = raw[raw.date.dt.year == 1996]
print("\n1996 READING DATES (Hejjala):")
print(r96[r96.station_name == "Hejjala"].sort_values("date")[["date", "data_value"]].to_string())

# 3) which months do readings fall in?
tidy = pd.read_csv("data/processed/groundwater/gw_tidy.csv", parse_dates=["date"])
print("\nREADINGS BY MONTH:")
print(tidy.date.dt.month.value_counts().sort_index().to_string())

# 4) season label from nearest nominal CGWB month
season_by_month = {1: "Jan", 2: "Jan", 3: "May", 4: "May", 5: "May", 6: "Aug",
                   7: "Aug", 8: "Aug", 9: "Nov", 10: "Nov", 11: "Nov", 12: "Jan"}
tidy["season"] = tidy.date.dt.month.map(season_by_month)
tidy["year"] = tidy.date.dt.year

seasonal = (tidy.groupby(["station_id", "station_name", "well_type", "lat", "lon",
                          "year", "season"], as_index=False)
            .depth_to_water_m.mean())
seasonal.to_csv("data/processed/groundwater/gw_seasonal.csv", index=False)
print("\nseasonal rows:", len(seasonal))

# 5) recent coverage, 2014 to 2024
recent = seasonal[seasonal.year >= 2014]
print("\nRECENT (2014+) SEASONAL READINGS PER WELL:")
print(recent.groupby("station_name").size().sort_values(ascending=False).to_string())
