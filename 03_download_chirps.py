"""
03_download_chirps.py  (v2 -- via Google Earth Engine, not direct global-file download)

Downloads CHIRPS v2.0 daily precipitation (Climate Hazards Center, UCSB) for
the study districts/events as an independent cross-check against ERA5-Land
precipitation.

WHAT CHANGED FROM v1:
  v1 downloaded the entire global daily grid as one ~1.5 GB NetCDF file per
  year (10 GB+ total) directly from UCSB's server, which is slow and prone
  to timing out on an ordinary connection -- exactly what happened. This
  version pulls the same data through Google Earth Engine's
  'UCSB-CHG/CHIRPS/DAILY' collection, requesting only each district's small
  bounding box (the same approach 06_download_sentinel_gee.py already uses
  for Sentinel-1/2), which is orders of magnitude less data and does not
  require downloading a raw global file at all.

  Every (district, day) value is cached incrementally per district-year to
  01_raw_data/chirps/, so if this script is interrupted (as it was), simply
  re-running it resumes from the next unfinished district-year instead of
  re-downloading anything already saved.

Requires the same Earth Engine project/authentication as
06_download_sentinel_gee.py -- see 00_setup_check.py. Set EE_PROJECT before
running, e.g.:
    import os; os.environ['EE_PROJECT'] = 'your-project-id'

Reference: Funk et al. (2015), Scientific Data 2, 150066.
"""
import sys
from pathlib import Path

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, DISTRICT_BBOXES, EVENT_WINDOWS, RAW, log_step

OUT_DIR = RAW / "chirps"
COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"


def district_geometry_ee(name: str):
    import ee
    try:
        import geopandas as gpd
        boundary_path = RAW / "boundaries" / "study_districts.geojson"
        if boundary_path.exists():
            gdf = gpd.read_file(boundary_path)
            name_col = [c for c in gdf.columns if c.upper().startswith("NAME")][-1]
            match = gdf[gdf[name_col] == name]
            if len(match) == 1:
                return ee.Geometry(match.geometry.iloc[0].__geo_interface__)
    except Exception:
        pass
    lon_min, lat_min, lon_max, lat_max = DISTRICT_BBOXES[name]
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def download_one_district_year(district, year, start, end, geom):
    """Returns a DataFrame of daily district-mean precip for one
    district-year, cached to disk so a repeat run skips it entirely."""
    import ee
    import pandas as pd

    target = OUT_DIR / f"chirps_{district}_{year}.csv"
    if target.exists() and target.stat().st_size > 0:
        try:
            cached = pd.read_csv(target)
            if not cached.empty:
                print(f"    {target.name} already exists ({len(cached)} rows), skipping.")
                return cached
        except Exception:
            pass  # fall through and re-fetch if the cached file is unreadable

    print(f"    Requesting CHIRPS daily precip: {district} {year} ({start}:{end}) ...")
    coll = (ee.ImageCollection(COLLECTION_ID)
            .filterBounds(geom)
            .filterDate(start, end))

    n_images = coll.size().getInfo()
    if n_images == 0:
        print(f"    WARNING: no CHIRPS images found for {district} {year}.")
        empty = pd.DataFrame(columns=["date", "precip_mm", "district", "event_year"])
        empty.to_csv(target, index=False)
        return empty

    def reduce_one(image):
        stats = image.select("precipitation").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=5000, maxPixels=1e9)
        return ee.Feature(None, stats.set("date", image.date().format("YYYY-MM-dd")))

    feats = coll.map(reduce_one).getInfo()["features"]
    rows = []
    for feat in feats:
        p = feat["properties"]
        rows.append({
            "date": p.get("date"),
            "precip_mm": p.get("precipitation"),
            "district": district,
            "event_year": year,
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print(f"    -> Saved {len(df)} days -> {target.name}")
    return df


def main():
    import ee
    import pandas as pd

    ee_project = os.environ.get("EE_PROJECT")
    if not ee_project:
        print("ERROR: no Earth Engine Cloud project set. Set it via the EE_PROJECT "
              "environment variable before running this script, e.g.\n"
              "    import os; os.environ['EE_PROJECT'] = 'your-project-id'")
        sys.exit(1)
    ee.Initialize(project=ee_project)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for district in DISTRICTS:
        geom = district_geometry_ee(district)
        print(f"DISTRICT: {district}")
        for year, (start, end) in EVENT_WINDOWS.items():
            df = download_one_district_year(district, year, start, end, geom)
            if not df.empty:
                all_rows.append(df)

        log_step(
            script="03_download_chirps.py", dataset="CHIRPS precipitation (cross-check)",
            provider="Climate Hazards Center, UCSB via Google Earth Engine",
            product_version=f"{COLLECTION_ID}", resolution="~5.5km, daily", units="mm/day",
            query_parameters=f"district={district}; all 5 event windows",
            output_path=str(OUT_DIR), rows_or_files="see console",
            status="OK", notes="via GEE (lighter-weight than direct global-file download)",
        )

    if all_rows:
        out_csv = RAW.parent / "02_processed_data" / "chirps_district_daily.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved combined CHIRPS district-daily table -> {out_csv} "
              f"({len(combined)} rows)")
    else:
        print("No CHIRPS data downloaded -- see warnings above.")


if __name__ == "__main__":
    main()
