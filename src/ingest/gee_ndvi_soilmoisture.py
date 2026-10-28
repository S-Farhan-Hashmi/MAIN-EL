"""
gee_ndvi_soilmoisture.py
------------------------
Extracts NDVI (Sentinel-2 + MODIS gap-fill) and ERA5-Land soil-moisture for
every locality polygon in the pilot corridor, then exports per-locality CSVs.

Pilot corridor bbox:  12.78-12.95 degN,  77.38-77.55 degE  (Bengaluru South fringe)

ERA5 note
---------
ERA5-Land has ~9-11 km (0.1 deg) resolution.  Locality polygons smaller than
one ERA5 pixel (LOC_01, LOC_02 at ~1.6-1.8 km²) return null from reduceRegions
because no pixel centroid falls inside the polygon.  This script works around
that by always sampling ERA5 at the locality centroid point (guaranteed to hit
a pixel), and falls back to polygon-mean for larger localities where it is
available.  The sampling method used is recorded in the `sm_method` column.

Usage
-----
  # Authenticate once (opens a browser):
  earthengine authenticate

  # Then run for a one-month test:
  python src/ingest/gee_ndvi_soilmoisture.py --start 2024-06-01 --end 2024-06-30

  # Optional flags:
  #   --out_dir     path to write CSVs (default: data/raw)
  #   --watersheds  path to local GeoJSON with watershed polygons
  #                 (default: gis/pilot_watersheds.geojson)
  #   --cloud_pct   max Sentinel-2 scene cloud cover allowed (default: 60)

Dependencies (add to environment.yml):
  earthengine-api >= 0.1.390
  geopandas
  pandas
  shapely
  tqdm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Pilot corridor defaults
# ---------------------------------------------------------------------------
CORRIDOR_BBOX = [77.38, 12.78, 77.55, 12.95]   # [west, south, east, north]
DEFAULT_CLOUD_PCT = 60
DEFAULT_SCALE_S2 = 10        # metres
DEFAULT_SCALE_MODIS = 250    # metres
DEFAULT_SCALE_ERA5 = 11132   # metres (~0.1 deg)


def bbox_to_ee_geometry(bbox: list[float]) -> ee.Geometry.Rectangle:
    west, south, east, north = bbox
    return ee.Geometry.Rectangle([west, south, east, north])


# ---------------------------------------------------------------------------
# NDVI from Sentinel-2 with cloud masking
# ---------------------------------------------------------------------------
def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    # divide() and updateMask() strip metadata; copy all properties back
    return (image.updateMask(mask).divide(10000)
            .copyProperties(image, image.propertyNames()))


def get_s2_ndvi(
    geometry: ee.Geometry, start: str, end: str,
    cloud_pct: int = DEFAULT_CLOUD_PCT
) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_s2_clouds)
        .map(lambda img: img
             .normalizedDifference(["B8", "B4"])
             .rename("NDVI_S2")
             .set("system:time_start", img.get("system:time_start")))
    )


# ---------------------------------------------------------------------------
# NDVI gap-fill from MODIS MOD13Q1 (16-day composites)
# ---------------------------------------------------------------------------
def get_modis_ndvi(
    geometry: ee.Geometry, start: str, end: str
) -> ee.ImageCollection:
    return (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start, end)
        .filterBounds(geometry)
        .map(lambda img: img.select("NDVI")
             .multiply(0.0001)
             .rename("NDVI_MODIS")
             .set("system:time_start", img.get("system:time_start")))
    )


# ---------------------------------------------------------------------------
# Soil moisture from ERA5-Land (volumetric, layer 1: 0-7 cm)
# ---------------------------------------------------------------------------
def get_era5_soilmoisture(
    geometry: ee.Geometry, start: str, end: str
) -> ee.ImageCollection:
    return (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .filterBounds(geometry)
        .select("volumetric_soil_water_layer_1")
        .map(lambda img: img.rename("SM_ERA5")
             .set("system:time_start", img.get("system:time_start")))
    )


# ---------------------------------------------------------------------------
# Reduce an ImageCollection over each watershed / locality polygon
# ---------------------------------------------------------------------------
def reduce_collection_over_watersheds(
    collection: ee.ImageCollection,
    watersheds_fc: ee.FeatureCollection,
    band_name: str,
    scale: int,
    reducer: ee.Reducer | None = None,
) -> ee.FeatureCollection:
    if reducer is None:
        reducer = ee.Reducer.mean()

    def reduce_image(image: ee.Image) -> ee.FeatureCollection:
        date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        reduced = image.select(band_name).reduceRegions(
            collection=watersheds_fc,
            reducer=reducer.setOutputs([band_name]),
            scale=scale,
        )
        return reduced.map(lambda f: f.set("date", date_str))

    return collection.map(reduce_image).flatten()


# ---------------------------------------------------------------------------
# ERA5 extractor with centroid-point fallback
# ---------------------------------------------------------------------------
def reduce_era5_over_localities(
    collection: ee.ImageCollection,
    gdf,                          # geopandas GeoDataFrame
    band_name: str,
    scale: int,
) -> pd.DataFrame:
    """
    For each locality, sample ERA5 at its centroid point (always returns a
    value, even for polygons smaller than one ERA5 pixel).  Also attempt a
    polygon-mean; where the polygon-mean is non-null it takes precedence.
    Records which method was used in the `sm_method` column.
    """
    import shapely.geometry as sg

    # Build a FeatureCollection of centroid points
    centroid_features = []
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        centroid_features.append(
            ee.Feature(
                ee.Geometry.Point([centroid.x, centroid.y]),
                {"watershed_id": row["watershed_id"]},
            )
        )
    centroid_fc = ee.FeatureCollection(centroid_features)

    # Build a FeatureCollection of polygons (for polygon-mean attempt)
    poly_features = [
        ee.Feature(
            ee.Geometry.Polygon(list(row.geometry.__geo_interface__["coordinates"])),
            {"watershed_id": row["watershed_id"]},
        )
        for _, row in gdf.iterrows()
    ]
    poly_fc = ee.FeatureCollection(poly_features)

    all_rows: list[dict] = []

    images = collection.toList(collection.size())
    n = collection.size().getInfo()
    for i in range(n):
        img = ee.Image(images.get(i))
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        img_band = img.select(band_name)

        # --- centroid sample (always works) ---
        sampled = img_band.sampleRegions(
            collection=centroid_fc,
            scale=scale,
            geometries=False,
        ).getInfo()["features"]
        centroid_vals = {
            f["properties"]["watershed_id"]: f["properties"].get(band_name)
            for f in sampled
        }

        # --- polygon mean (may return null for tiny polygons) ---
        reduced = img_band.reduceRegions(
            collection=poly_fc,
            reducer=ee.Reducer.mean().setOutputs([band_name]),
            scale=scale,
        ).getInfo()["features"]
        poly_vals = {
            f["properties"]["watershed_id"]: f["properties"].get(band_name)
            for f in reduced
        }

        for _, row in gdf.iterrows():
            wid = row["watershed_id"]
            poly_v = poly_vals.get(wid)
            cent_v = centroid_vals.get(wid)
            if poly_v is not None:
                sm_val = poly_v
                method = "polygon_mean"
            elif cent_v is not None:
                sm_val = cent_v
                method = "centroid_point"
            else:
                sm_val = None
                method = "missing"
            all_rows.append({
                "watershed_id": wid,
                "date": date_str,
                band_name: sm_val,
                "sm_method": method,
            })

    return pd.DataFrame(all_rows)


def fc_to_dataframe(fc: ee.FeatureCollection, columns: list[str]) -> pd.DataFrame:
    features = fc.select(columns).getInfo()["features"]
    rows = [f["properties"] for f in features]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(start: str, end: str, out_dir: Path, watersheds_path: Path,
        cloud_pct: int) -> None:
    try:
        ee.Initialize(project=os.environ.get("GEE_PROJECT", "excellent-hue-480109-m8"))
    except ee.EEException:
        print("[INFO] Triggering interactive authentication...")
        ee.Authenticate()
        ee.Initialize(project=os.environ.get("GEE_PROJECT", "ee-pilot-corridor"))

    bbox_geom = bbox_to_ee_geometry(CORRIDOR_BBOX)

    if watersheds_path.exists():
        gdf = gpd.read_file(watersheds_path)
    else:
        print(f"[WARN] {watersheds_path} not found - using corridor bbox as single polygon.")
        import shapely.geometry as sg
        west, south, east, north = CORRIDOR_BBOX
        gdf = gpd.GeoDataFrame(
            {"watershed_id": ["pilot_corridor"]},
            geometry=[sg.box(west, south, east, north)],
            crs="EPSG:4326",
        )

    watersheds_fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Polygon(list(row.geometry.__geo_interface__["coordinates"])),
            {"watershed_id": row["watershed_id"]},
        )
        for _, row in gdf.iterrows()
    ])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sentinel-2 NDVI
    print("[1/3] Fetching Sentinel-2 NDVI...")
    s2_fc = reduce_collection_over_watersheds(
        get_s2_ndvi(bbox_geom, start, end, cloud_pct),
        watersheds_fc, "NDVI_S2", DEFAULT_SCALE_S2,
    )
    s2_df = fc_to_dataframe(s2_fc, ["watershed_id", "date", "NDVI_S2"])
    s2_df["data_source"] = "Sentinel-2 SR"
    s2_df["gap_filled"] = False
    s2_path = out_dir / "ndvi_sentinel2.csv"
    s2_df.to_csv(s2_path, index=False)
    print(f"    -> {s2_path}  ({len(s2_df)} rows)")

    # MODIS NDVI
    print("[2/3] Fetching MODIS NDVI...")
    modis_fc = reduce_collection_over_watersheds(
        get_modis_ndvi(bbox_geom, start, end),
        watersheds_fc, "NDVI_MODIS", DEFAULT_SCALE_MODIS,
    )
    modis_df = fc_to_dataframe(modis_fc, ["watershed_id", "date", "NDVI_MODIS"])
    modis_df["data_source"] = "MODIS MOD13Q1"
    modis_df["gap_filled"] = True
    modis_path = out_dir / "ndvi_modis.csv"
    modis_df.to_csv(modis_path, index=False)
    print(f"    -> {modis_path}  ({len(modis_df)} rows)")

    # ERA5-Land soil moisture (centroid-fallback for tiny locality polygons)
    print("[3/3] Fetching ERA5-Land soil moisture (with centroid fallback for small polygons)...")
    era5_df = reduce_era5_over_localities(
        get_era5_soilmoisture(bbox_geom, start, end),
        gdf, "SM_ERA5", DEFAULT_SCALE_ERA5,
    )
    era5_df["data_source"] = "ERA5-Land daily"
    era5_df["note"] = ("Model-based reanalysis; polygon_mean used where polygon > ERA5 pixel, "
                       "centroid_point used for small locality polygons (LOC_01, LOC_02).")
    era5_path = out_dir / "soilmoisture_era5.csv"
    era5_df.to_csv(era5_path, index=False)
    missing = era5_df["SM_ERA5"].isna().sum()
    print(f"    -> {era5_path}  ({len(era5_df)} rows, {missing} still missing)")
    if missing:
        print(f"    [WARN] {missing} rows still null after centroid fallback.")
    method_counts = era5_df["sm_method"].value_counts().to_dict()
    print(f"    Sampling methods used: {method_counts}")

    print("\n[DONE] All CSVs written to", out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--out_dir", default="data/raw",
                   help="Output directory for CSVs (default: data/raw)")
    p.add_argument("--watersheds", default="gis/pilot_watersheds.geojson",
                   help="Watershed GeoJSON path (default: gis/pilot_watersheds.geojson)")
    p.add_argument("--cloud_pct", type=int, default=DEFAULT_CLOUD_PCT,
                   help=f"Max Sentinel-2 cloud %% (default: {DEFAULT_CLOUD_PCT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=args.start,
        end=args.end,
        out_dir=Path(args.out_dir),
        watersheds_path=Path(args.watersheds),
        cloud_pct=args.cloud_pct,
    )
