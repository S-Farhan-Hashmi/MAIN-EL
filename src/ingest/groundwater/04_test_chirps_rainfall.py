from pathlib import Path
import gzip
import requests
import rasterio
import pandas as pd

ROOT = Path.cwd()
TEST_DIR = ROOT / "data/raw/rainfall/test"
OUT_DIR = ROOT / "data/interim/groundwater"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the station coordinates from the cleaned groundwater dataset.
STATIONS = (
    ROOT / "data/processed/groundwater/gw_tidy.csv"
)
MONTHS = [1, 5, 8, 11]
YEAR = 2024

df = pd.read_csv(STATIONS)
station_cols = ["station_id", "station_name", "latitude", "longitude"]
print("Available columns:", df.columns.tolist())

# Match the actual column names in the dataset.
lat_col = next(c for c in ["latitude", "lat"] if c in df.columns)
lon_col = next(c for c in ["longitude", "lon", "long"] if c in df.columns)
id_col = next(c for c in ["station_id"] if c in df.columns)
name_col = next(c for c in ["station_name", "station"] if c in df.columns)

stations = df[[id_col, name_col, lat_col, lon_col]].drop_duplicates(id_col)
rows = []

for month in MONTHS:
    filename = f"chirps-v2.0.{YEAR}.{month:02d}.tif"
    tif_path = TEST_DIR / filename
    gz_path = TEST_DIR / f"{filename}.gz"

    if not tif_path.exists():
        url = (
            "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
            f"global_monthly/tifs/{filename}.gz"
        )
        print("Downloading:", url)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        gz_path.write_bytes(response.content)

        with gzip.open(gz_path, "rb") as src, open(tif_path, "wb") as dst:
            dst.write(src.read())

    with rasterio.open(tif_path) as raster:
        coords = list(zip(stations[lon_col], stations[lat_col]))
        values = list(raster.sample(coords))

        for (_, station), sample in zip(stations.iterrows(), values):
            value = float(sample[0])
            rows.append({
                "station_id": station[id_col],
                "station_name": station[name_col],
                "year": YEAR,
                "month": month,
                "rainfall_value": value,
                "source": "CHIRPS v2.0 global monthly GeoTIFF",
                "source_unit": "verify from product documentation",
                "is_imputed": False,
            })

out = pd.DataFrame(rows)
out_path = OUT_DIR / "chirps_test_2024.csv"
out.to_csv(out_path, index=False)

print("\nRainfall samples:")
print(out.to_string(index=False))
print("\nSaved:", out_path)
