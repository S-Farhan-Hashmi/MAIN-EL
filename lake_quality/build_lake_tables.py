#!/usr/bin/env python3
"""
Build the lake-model tables from KSPCB monthly lake PDFs.

Steps
  1. Extract the table from every monthly PDF (training months + test month)
  2. Clean: coordinates, BDL/NA values, month/year, lake_id
  3. Stack training months into one table, keep the test month separate
  4. Add population within 1 km and 2 km of each lake from the WorldPop raster

Usage
  python build_lake_tables.py --train_dir pdfs_train --test_pdf Bengaluru_Lakes_July_-2026.pdf \
      --pop_raster ind_ppp_2020_constrained.tif --out_dir out

Filenames must contain the year: either 4 digits (Lakes_Dec-2024.pdf) or month + 2 digits (LAKE_AUG_25.pdf).
The month is read from the 'Sampling Month' column inside the PDF.
--pop_raster is optional; leave it out to skip the population step.
"""
import argparse
import difflib
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pdfplumber

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# output column name -> regex matched against the cleaned PDF header
KEY_COLS = {
    "temperature_c": r"^temperat",
    "do_mgl": r"^dissolved o2",
    "ph": r"^ph$",
    "conductivity": r"^conductivity",
    "bod_mgl": r"^bod",
    "nitrate_n": r"^nitrate",
    "nitrite_n": r"^nitrite",
    "fecal_coliform": r"^fecal coliform",
    "total_coliform": r"^total coliform",
    "turbidity_ntu": r"^turbidity",
    "cod_mgl": r"^cod",
    "tkn": r"^total kjeldahl",
    "ammonia_n": r"^ammonic",
    "chloride": r"^chlorides",
    "tds": r"^total dissolved",
    "tss": r"^total suspended",
    "total_phosphate": r"^(total )?phosphate",
    "ortho_phosphate": r"^ortho phosphate",
}

# Detection limits taken from the footnote of the KSPCB report (assumption: see README).
# BDL is replaced by half of this value; where no limit is known the value becomes NaN.
DETECTION_LIMIT = {
    "do_mgl": 0.3, "nitrite_n": 0.02, "tkn": 1.5, "ammonia_n": 0.4,
    "tss": 10.0, "total_phosphate": 0.2, "ortho_phosphate": 0.05,
}

# rough box around Bengaluru + the Kengeri-Bidadi belt; outside it a coordinate is flagged
LAT_RANGE = (12.60, 13.25)
LON_RANGE = (77.25, 77.90)


def clean_header(h):
    return re.sub(r"\s+", " ", str(h or "").replace("\n", " ")).strip().lower()


def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "").replace("\n", " ")).strip()


def to_float_coord(s):
    # some rows come as '12.88 6957' (a stray space) or with quotes
    s = str(s or "").replace('"', "").replace(" ", "").replace("\n", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def find_col(header, pattern):
    return next((i for i, h in enumerate(header) if re.search(pattern, h)), None)


def parse_year(fname):
    m = re.search(r"(20\d\d)", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_ .]*(\d{2})(?!\d)", fname.lower())
    if m:
        return 2000 + int(m.group(1))
    return np.nan


def extract_pdf(path):
    year = parse_year(os.path.basename(path))
    if np.isnan(year):
        print(f"  WARNING: no year in filename {os.path.basename(path)}", file=sys.stderr)

    header, raw_rows, preview = None, [], []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if pno == 0:
                preview.append(f"page 1 has {len(tables)} table(s)")
            for table in tables:
                for r in table:
                    if not r:
                        continue
                    cells = [clean_header(c) for c in r]
                    if len(preview) < 4 and pno == 0:
                        preview.append(str(cells[:6]))
                    # the header row is the one that names the location / sampling month columns
                    if header is None and any(
                            re.sub(r"\s", "", c).startswith("samplingmonth") or "nameofmonitoring" in re.sub(r"\s", "", c)
                            for c in cells):
                        header = cells
                        continue
                    if header is not None and len(r) > 20:
                        raw_rows.append(r)
    if header is None:
        raise RuntimeError("No header row found. What the script saw: " + " | ".join(preview))

    id_is_stn = header[0].startswith("stn")
    i_month = next((i for i, h in enumerate(header)
                    if re.sub(r"\s", "", h).startswith("samplingmonth")), None)   # some PDFs split 'samplin g month'
    i_name = find_col(header, r"name of monitoring|^location|^station name")
    if i_name is None and i_month is not None:
        i_name = i_month + 1
    i_lat = find_col(header, r"^latitude")
    i_lon = find_col(header, r"^longitude")
    i_use = find_col(header, r"^use based")

    def get(r, i):
        return clean_text(r[i]) if i is not None and i < len(r) else ""

    out = []
    for r in raw_rows:
        month_txt = get(r, i_month)[:3].lower()
        if month_txt not in MONTHS or not get(r, i_name):
            continue                      # repeated header rows, notes, blanks
        first_cell = str(r[0] or "").strip()
        first = int(first_cell) if first_cell.isdigit() else np.nan
        rec = {
            "source_file": os.path.basename(path),
            "year": year,
            "stn_code": first if id_is_stn else np.nan,
            "sl_no": np.nan if id_is_stn else first,
            "month_name": month_txt,
            "lake_name": get(r, i_name),
            "lat": to_float_coord(get(r, i_lat)),
            "lon": to_float_coord(get(r, i_lon)),
            "use_class": get(r, i_use),
        }
        for out_name, pattern in KEY_COLS.items():
            rec[out_name + "_raw"] = get(r, find_col(header, pattern))
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        raise RuntimeError("Header found but no data rows matched. Header seen: " + str(header[:8]))
    df["month"] = df["month_name"].map(MONTHS)
    has_coords = "yes" if i_lat is not None else "NO (will be filled from other files)"
    print(f"  {os.path.basename(path)}: {len(df)} rows, id column = "
          f"{'STN code' if id_is_stn else 'Sl.No'}, coordinates in PDF: {has_coords}")
    return df


def clean_values(df):
    for col in KEY_COLS:
        raw = df[col + "_raw"].str.upper()
        bdl = raw.eq("BDL")
        vals = pd.to_numeric(df[col + "_raw"], errors="coerce")
        if col in DETECTION_LIMIT:
            vals = vals.where(~bdl, DETECTION_LIMIT[col] / 2)
        df[col] = vals
        df[col + "_bdl"] = bdl
        df = df.drop(columns=[col + "_raw"])
    return df


def name_key(name):
    s = re.sub(r"\(.*?\)", " ", name)       # drop parenthetical text
    s = s.split(",")[0]                      # drop address after the first comma
    return re.sub(r"[^a-z0-9]", "", s.lower())


MERGE_LOG = []          # filled by canonicalize_keys, written to lake_id_merges.csv


def canonicalize_keys(df, threshold=0.85):
    """Some monthly PDFs lose letters inside names ('Benniganahalli' -> 'bennigahalli',
    'Nallurahalli' -> 'llurahalli'). Map such variants onto the correctly spelled key.
    Reference spellings = names from rows that carry coordinates (the July 2026 file).
    Guard: two keys that appear in the same monthly file are different lakes, never merged."""
    has = df["lat"].notna() & df["lon"].notna()
    files = df.groupby("lake_key")["source_file"].agg(set).to_dict()
    ref = set(df.loc[has, "lake_key"])
    canon = set(ref)
    mapping = {}
    others = sorted((k for k in files if k not in ref and k), key=lambda k: -len(files[k]))
    for k in others:
        best, best_r = None, 0.0
        for c in canon:
            if files[c] & files[k]:
                continue
            r = difflib.SequenceMatcher(None, k, c).ratio()
            if r > best_r:
                best, best_r = c, r
        if best is not None and best_r >= threshold:
            mapping[k] = best
            files[best] = files[best] | files[k]
            MERGE_LOG.append({"variant": k, "merged_into": best,
                              "similarity": round(best_r, 3), "n_files": len(files[k])})
        else:
            canon.add(k)
    df = df.copy()
    df["lake_key"] = df["lake_key"].map(lambda k: mapping.get(k, k))
    return df


def assign_lake_ids(df):
    """lake_id comes from the cleaned name. Same name with far-apart coordinates
    (two different 'Puttenahalli Lake's) is split by coordinate cluster.
    Rows with no coordinates (some months omit them) borrow coordinates from other
    months by name, but only if the name maps to exactly one place."""
    df = df.copy()
    df["lake_key"] = df["lake_name"].map(name_key)
    df = canonicalize_keys(df)
    df["lake_id"] = df["lake_key"]
    has = df["lat"].notna() & df["lon"].notna()
    df["coord_source"] = np.where(has, "pdf", "")

    centroids_by_key = {}
    for key, g in df[has].groupby("lake_key"):
        cents, assigned = [], {}
        for idx, row in g.iterrows():
            k = next((i for i, (a, b) in enumerate(cents)
                      if abs(row["lat"] - a) < 0.03 and abs(row["lon"] - b) < 0.03), None)
            if k is None:
                cents.append((row["lat"], row["lon"]))
                k = len(cents) - 1
            assigned[idx] = k
        if len(cents) > 1:
            for idx, k in assigned.items():
                df.at[idx, "lake_id"] = f"{key}_{k + 1}"
        centroids_by_key[key] = cents

    for idx in df.index[~has]:
        key = df.at[idx, "lake_key"]
        cents = centroids_by_key.get(key, [])
        if len(cents) == 1:
            df.at[idx, "lat"], df.at[idx, "lon"] = cents[0]
            df.at[idx, "coord_source"] = "lookup"
        elif len(cents) > 1:
            df.at[idx, "lake_id"] = key + "_ambiguous"
            df.at[idx, "coord_source"] = "ambiguous"
        else:
            df.at[idx, "coord_source"] = "missing"
    return df.drop(columns="lake_key")


def buffer_population(raster_path, lats, lons, radius_m):
    import rasterio
    from rasterio.windows import Window

    out = []
    with rasterio.open(raster_path) as src:
        nod = src.nodata
        for la, lo in zip(lats, lons):
            if np.isnan(la) or np.isnan(lo):
                out.append(np.nan)
                continue
            dlat = radius_m / 111320.0
            dlon = radius_m / (111320.0 * np.cos(np.radians(la)))
            win = src.window(lo - dlon, la - dlat, lo + dlon, la + dlat)
            win = Window(int(np.floor(win.col_off)), int(np.floor(win.row_off)),
                         int(np.ceil(win.width)) + 1, int(np.ceil(win.height)) + 1)
            arr = src.read(1, window=win, boundless=True, fill_value=nod if nod is not None else 0)
            tr = src.window_transform(win)
            rows, cols = np.indices(arr.shape)
            xs, ys = tr * (cols + 0.5, rows + 0.5)
            dx = (xs - lo) * 111320.0 * np.cos(np.radians(la))
            dy = (ys - la) * 111320.0
            inside = (dx ** 2 + dy ** 2) <= radius_m ** 2
            valid = (arr >= 0) if nod is None else ((arr != nod) & (arr >= 0))
            out.append(float(arr[inside & valid].sum()))
    return out


def add_population(df, raster_path):
    lakes = df.groupby("lake_id")[["lat", "lon"]].median()
    for r in (1000, 2000):
        lakes[f"pop_{r // 1000}km"] = buffer_population(raster_path, lakes["lat"], lakes["lon"], r)
    return df.merge(lakes[["pop_1km", "pop_2km"]], left_on="lake_id", right_index=True, how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_pdf", required=True)
    ap.add_argument("--pop_raster")
    ap.add_argument("--coord_overrides",
                    help="CSV with columns lake_id,lat,lon to fill coordinates by hand "
                         "(copy out/coords_needed.csv, fill lat/lon, pass the copy here)")
    ap.add_argument("--out_dir", default="out")
    a = ap.parse_args()
    if not os.path.isdir(a.train_dir):
        sys.exit(f"--train_dir folder not found: {a.train_dir}")
    if not os.path.isfile(a.test_pdf):
        sys.exit(f"--test_pdf file not found: {a.test_pdf}\n"
                 "Check the exact filename (run 'dir' in this folder) or give the full path in quotes.")
    os.makedirs(a.out_dir, exist_ok=True)

    print("Extracting training PDFs")
    train_files = sorted(glob.glob(os.path.join(a.train_dir, "*.pdf")))
    if not train_files:
        sys.exit("No PDFs found in --train_dir")
    parts, failed = [], []
    for f in train_files:
        try:
            parts.append(extract_pdf(f))
        except Exception as e:                      # keep going, report at the end
            failed.append((os.path.basename(f), str(e)))
            print(f"  FAILED: {os.path.basename(f)}", file=sys.stderr)
    if failed:
        print("\n*** These files were SKIPPED, send them for a fix: ***")
        for name, why in failed:
            print(f"  - {name}: {why}")
        print()
    if not parts:
        sys.exit("No training PDFs could be read")
    train = pd.concat(parts, ignore_index=True)
    train["split"] = "train"
    print("Extracting test PDF")
    test = extract_pdf(a.test_pdf)
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df = clean_values(df)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    df = assign_lake_ids(df)
    if a.coord_overrides:
        ov = pd.read_csv(a.coord_overrides).dropna(subset=["lake_id", "lat", "lon"])
        ov = ov.drop_duplicates("lake_id").set_index("lake_id")
        m = df["lake_id"].isin(ov.index) & ~df["lake_id"].str.endswith("_ambiguous")
        df.loc[m, "lat"] = df.loc[m, "lake_id"].map(ov["lat"]).astype(float)
        df.loc[m, "lon"] = df.loc[m, "lake_id"].map(ov["lon"]).astype(float)
        df.loc[m, "coord_source"] = "manual"
        print(f"  manual coordinates applied to {int(m.sum())} rows ({ov.index.isin(df.lake_id).sum()} lakes)")
    df["coord_flag"] = ~(df["lat"].between(*LAT_RANGE) & df["lon"].between(*LON_RANGE))

    if a.pop_raster:
        print("Computing buffer population")
        df = add_population(df, a.pop_raster)

    first = ["split", "date", "year", "month", "stn_code", "lake_id", "lake_name", "lat", "lon",
             "coord_source", "use_class", "coord_flag"]
    df = df[first + [c for c in df.columns if c not in first]]

    df[df.split == "train"].drop(columns="split").to_csv(os.path.join(a.out_dir, "lake_training.csv"), index=False)
    df[df.split == "test"].drop(columns="split").to_csv(os.path.join(a.out_dir, "lake_test.csv"), index=False)

    # diagnostics
    months_per_lake = df[df.split == "train"].groupby("lake_id")["date"].nunique()
    print("\nSummary")
    print(f"  training rows: {(df.split == 'train').sum()}  test rows: {(df.split == 'test').sum()}")
    print(f"  distinct lake_ids: {df.lake_id.nunique()}")
    print(f"  lakes seen in <6 training months: {(months_per_lake < 6).sum()} (check name spelling)")
    print(f"  rows with flagged coordinates: {int(df.coord_flag.sum())}")
    print("  coordinate source:", df.coord_source.value_counts().to_dict())
    need = (df[df.coord_source == "missing"].groupby("lake_id")
            .agg(lake_name=("lake_name", "first"), n_rows=("lake_name", "size")).reset_index())
    need["lat"] = ""
    need["lon"] = ""
    need.to_csv(os.path.join(a.out_dir, "coords_needed.csv"), index=False)
    print(f"  lakes still missing coordinates: {len(need)} (fill out/coords_needed.csv, see --coord_overrides)")
    print(f"  misspelled name variants merged: {len(MERGE_LOG)} (review out/lake_id_merges.csv)")
    pd.DataFrame(MERGE_LOG, columns=["variant", "merged_into", "similarity", "n_files"]).to_csv(
        os.path.join(a.out_dir, "lake_id_merges.csv"), index=False)
    print(f"  training months: {sorted(df[df.split == 'train'].date.dropna().dt.strftime('%Y-%m').unique())}")
    df[df.coord_flag][["split", "date", "lake_name", "lat", "lon", "coord_source"]].drop_duplicates(
        subset=["lake_name"]).to_csv(os.path.join(a.out_dir, "flagged_coordinates.csv"), index=False)


if __name__ == "__main__":
    main()
