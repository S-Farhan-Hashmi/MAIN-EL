# Data Sources

**Project:** MAIN-EL – Micro-watershed AI-based Irrigation & Nutrient Enhancement Lab  
**Pilot Corridor:** 12.78–12.95°N, 77.38–77.55°E (Bengaluru South fringe, Karnataka)  
**Pilot Period:** June 2024 (one-month test run)  
**Spatial unit:** Locality-level polygons (8 localities)  
**Document last updated:** 2026-10-27  

---

## 1. Pilot Localities

All data is now collected and stored at the **locality level** (not macro-watershed level).
Each locality has its own polygon in `gis/pilot_watersheds.geojson`.

| Locality ID | Name | Full Name | Parent Basin | Area (km²) | Centroid (lat, lon) |
|---|---|---|---|---|---|
| LOC_01 | Bang_Uni_Ars_Ls | Bangalore University Arts Lower Section | Bengaluru_South_Basin | 1.8 | 12.9265°N, 77.5015°E |
| LOC_02 | Bang_Uni_Ars_Us | Bangalore University Arts Upper Section | Bengaluru_South_Basin | 1.6 | 12.9310°N, 77.5060°E |
| LOC_03 | Suggaunahalli | Suggaunahalli | Bengaluru_South_Basin | 3.2 | 12.9175°N, 77.4650°E |
| LOC_04 | Bidadi | Bidadi | Kanakapura_Basin | 18.5 | 12.7985°N, 77.3920°E |
| LOC_05 | Hejjala | Hejjala | Kanakapura_Basin | 8.4 | 12.8325°N, 77.4020°E |
| LOC_06 | Thalaghattapura | Thalaghattapura | Bengaluru_South_Basin | 6.7 | 12.8780°N, 77.5160°E |
| LOC_07 | Kengeri | Kengeri | Bengaluru_South_Basin | 14.2 | 12.9120°N, 77.4830°E |
| LOC_08 | Jnanabharathi | Jnanabharathi | Bengaluru_South_Basin | 4.9 | 12.9335°N, 77.5055°E |

> **Note:** LOC_04 (Bidadi) and LOC_05 (Hejjala) fall within the Kanakapura Basin (rural/agricultural).
> All other localities fall within the Bengaluru South Basin (peri-urban/urban fringe).

---

## 2. Locality Boundaries

| Field | Detail |
|---|---|
| **Source** | Manually delineated bounding boxes from OpenStreetMap locality centroids |
| **File in repo** | `gis/pilot_watersheds.geojson` |
| **Projection** | WGS 84 (EPSG:4326) |
| **Number of features** | 8 locality polygons |
| **Properties per feature** | `watershed_id`, `watershed_name`, `full_name`, `centroid_lat`, `centroid_lon`, `area_km2`, `parent_basin` |
| **How to update** | Replace bounding boxes with cadastral / village boundary shapefiles from Karnataka Revenue Dept or BBMP GIS when available |

---

## 3. NDVI – Sentinel-2 (Primary)

| Field | Detail |
|---|---|
| **Source** | Sentinel-2 Surface Reflectance, Harmonized |
| **Provider** | European Space Agency (ESA) / Copernicus |
| **GEE Collection ID** | `COPERNICUS/S2_SR_HARMONIZED` |
| **Access** | Free via Google Earth Engine |
| **Spatial resolution** | 10 m |
| **Temporal resolution** | ~5 days revisit; cloud-free scenes vary |
| **Bands used** | B8 (NIR), B4 (Red) → NDVI = (B8 – B4) / (B8 + B4) |
| **Cloud masking** | QA60 band – cloud bit (10) and cirrus bit (11) masked |
| **Cloud threshold** | Scenes with > 60% cloud cover excluded |
| **Scale factor** | Raw DN / 10,000 |
| **Output file** | `data/raw/ndvi_sentinel2.csv` |
| **Rows** | 16 (8 localities × 2 cloud-free scenes) |
| **Scenes available (June 2024)** | 2 cloud-free scenes: 2024-06-01, 2024-06-16 |
| **Limitation** | Heavy monsoon cloud cover causes large gaps June–September; expect only 1–3 usable scenes per month during SW monsoon |

### Columns in `ndvi_sentinel2.csv`

| Column | Description |
|---|---|
| `watershed_id` | Locality identifier (LOC_01 … LOC_08) |
| `date` | Overpass date (YYYY-MM-DD) |
| `NDVI_S2` | Mean NDVI over locality polygon; blank = cloud gap |
| `data_source` | `Sentinel-2 SR` |
| `gap_filled` | False for all rows in this file (filling done in feature pipeline) |

---

## 4. NDVI – MODIS (Gap-fill)

| Field | Detail |
|---|---|
| **Source** | MOD13Q1 v6.1 – 16-day NDVI composite |
| **Provider** | NASA LP DAAC |
| **GEE Collection ID** | `MODIS/061/MOD13Q1` |
| **Access** | Free via Google Earth Engine |
| **Spatial resolution** | 250 m |
| **Temporal resolution** | 16-day composites |
| **Band used** | `NDVI` |
| **Scale factor** | Raw value × 0.0001 |
| **Output file** | `data/raw/ndvi_modis.csv` |
| **Rows** | 16 (8 localities × 2 MODIS composites) |
| **Composites available (June 2024)** | 2 composites: 2024-06-09, 2024-06-25 |
| **Role** | Gap-fill only – used where Sentinel-2 is cloud-blocked |
| **Limitation** | 250 m resolution is coarser than the smaller locality polygons (LOC_01, LOC_02 are ~1–2 km²); MODIS pixel may extend beyond locality boundary |

### Columns in `ndvi_modis.csv`

| Column | Description |
|---|---|
| `watershed_id` | Locality identifier (LOC_01 … LOC_08) |
| `date` | Composite start date (YYYY-MM-DD) |
| `NDVI_MODIS` | Mean NDVI over locality polygon |
| `data_source` | `MODIS MOD13Q1` |
| `gap_filled` | True for all rows – this dataset is used only as gap-fill |

---

## 5. Soil Moisture – ERA5-Land (Primary)

| Field | Detail |
|---|---|
| **Source** | ERA5-Land Daily Aggregated v5 |
| **Provider** | ECMWF / Copernicus Climate Change Service |
| **GEE Collection ID** | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| **Access** | Free via Google Earth Engine (also via CDS API) |
| **Spatial resolution** | ~9–11 km (0.1° × 0.1°) |
| **Temporal resolution** | Daily |
| **Variable used** | `volumetric_soil_water_layer_1` (0–7 cm depth) |
| **Units** | m³/m³ (volumetric water content) |
| **Output file** | `data/raw/soilmoisture_era5.csv` |
| **Rows** | 232 (8 localities × 29 days) |
| **Days available (June 2024)** | 29 days: 2024-06-01 to 2024-06-29 |
| **Limitation** | Model-based reanalysis at ~10 km resolution. Localities smaller than one ERA5 pixel will share near-identical values unless locality-specific offsets are applied. Use ERA5 as a regional wetness driver; augment with IoT sensors for field-level accuracy. |

### Columns in `soilmoisture_era5.csv`

| Column | Description |
|---|---|
| `watershed_id` | Locality identifier (LOC_01 … LOC_08) |
| `date` | Date (YYYY-MM-DD) |
| `SM_ERA5` | Mean volumetric soil water (m³/m³) over locality polygon |
| `data_source` | `ERA5-Land daily` |
| `note` | Limitation reminder |

---

## 6. Extraction Script

| Field | Detail |
|---|---|
| **Script** | `src/ingest/gee_ndvi_soilmoisture.py` |
| **Language** | Python 3.11 |
| **Key dependency** | `earthengine-api >= 1.7` |
| **Authentication** | `earthengine authenticate` (one-time browser login) |
| **GEE Project used** | `excellent-hue-480109-m8` |
| **How to re-run** | `python src/ingest/gee_ndvi_soilmoisture.py --start YYYY-MM-DD --end YYYY-MM-DD` |
| **GeoJSON used** | `gis/pilot_watersheds.geojson` – now contains 8 locality polygons |

---

## 7. Feature Pipeline

| Field | Detail |
|---|---|
| **Script** | `src/features/build_features.py` |
| **Output** | `data/processed/master_features.csv` |
| **Rows** | 232 (8 localities × 29 days for June 2024 pilot) |
| **Columns** | 57 |
| **Key operations** | Daily grid alignment, forward-fill with flags, S2→MODIS gap-fill, lag/rolling features, seasonal flags, static proxy features |
| **How to run** | `python src/features/build_features.py` |

---

## 8. Static Proxy Features (per Locality)

These are survey/census estimates embedded in `build_features.py`. Replace with validated data when available.

| Locality ID | Name | Area km² | Borewell density | Irrigated % | Pop density | Dominant crop | Soil type |
|---|---|---|---|---|---|---|---|
| LOC_01 | Bang_Uni_Ars_Ls | 1.8 | 4.2 | 12.5 | 1850 | lawn_ornamental | red_laterite |
| LOC_02 | Bang_Uni_Ars_Us | 1.6 | 3.8 | 10.0 | 1620 | lawn_ornamental | red_laterite |
| LOC_03 | Suggaunahalli | 3.2 | 8.6 | 28.4 | 740 | vegetables_horticulture | red_laterite |
| LOC_04 | Bidadi | 18.5 | 14.2 | 45.8 | 320 | paddy_ragi | red_sandy_loam |
| LOC_05 | Hejjala | 8.4 | 11.9 | 40.1 | 280 | paddy_sugarcane | red_sandy_loam |
| LOC_06 | Thalaghattapura | 6.7 | 7.3 | 22.6 | 560 | vegetables_horticulture | black_cotton |
| LOC_07 | Kengeri | 14.2 | 5.1 | 15.3 | 2140 | vegetables_horticulture | black_cotton |
| LOC_08 | Jnanabharathi | 4.9 | 4.7 | 18.0 | 1200 | lawn_ornamental | red_laterite |

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Sentinel-2 gaps during monsoon (Jun–Sep) | NDVI missing for most of the irrigation-critical season | Forward-fill up to 5 days; MODIS gap-fill flagged in `NDVI_source` column |
| ERA5 soil moisture coarser than localities | Small localities (LOC_01, LOC_02 < 2 km²) share near-identical SM values | Use ERA5 as regional trend signal; IoT sensors for field-level moisture |
| MODIS at 250 m may exceed small locality polygons | NDVI gap-fill less accurate for LOC_01/LOC_02 | All MODIS-filled rows flagged with `NDVI_gap_filled=True` |
| Locality boundaries are approximate bboxes | Not aligned to cadastral village boundaries | Replace with Karnataka Revenue Dept / BBMP shapefiles when available |
| Proxy features (borewell_density etc.) are estimates | Static features not validated against ground truth | Replace with KSNDMC / Census data when available |
| One month of data only (June 2024) | Insufficient for robust model training | Extend date range using `--start` / `--end` flags in extraction script |

---

## 10. Data Sources to Check (Not Yet Integrated)

| Source | What it offers | Portal |
|---|---|---|
| KSNDMC | State-level soil moisture and agro-met data for Karnataka | [ksndmc.org](http://www.ksndmc.org) |
| ISRO Bhuvan / MOSDAC | Vegetation and soil moisture products at regional scale | [bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in) |
| SMAP (NASA) | 9–36 km satellite soil moisture, direct measurement | [nsidc.org/data/smap](https://nsidc.org/data/smap) |
| Sentinel-1 SAR | 10 m derived soil moisture (noisy over vegetation) | GEE: `COPERNICUS/S1_GRD` |
| Copernicus SWI | Soil Water Index, coarse regional scale | [land.copernicus.eu](https://land.copernicus.eu) |
| Karnataka Revenue Dept GIS | Cadastral village boundary shapefiles for precise locality polygons | Revenue Dept portal |

---

*All data extracted via Google Earth Engine using the script at `src/ingest/gee_ndvi_soilmoisture.py`.
Re-run with the updated `gis/pilot_watersheds.geojson` (8 locality polygons) and a wider date range to extend the dataset beyond the June 2024 pilot.*
