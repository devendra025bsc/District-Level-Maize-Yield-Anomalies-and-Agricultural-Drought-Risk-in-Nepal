"""
05_download_soilgrids.py

Downloads static soil properties (bdod, cec, clay, nitrogen, phh2o, sand,
silt, soc) from ISRIC SoilGrids v2.0 for the centroid of each study district,
via the public SoilGrids REST API. No authentication required.

Reference: Poggio et al. (2021), SOIL 7(1):217-240.

NOTE: SoilGrids properties are static (not per-event), so this script
produces one row per district, not per district-event. If district-mean
(rather than centroid-point) values are required for the manuscript, replace
the point query below with a WCS raster clip over the district polygon from
01_get_boundaries.py.
"""
import sys
import time
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, DISTRICT_BBOXES, RAW, log_step

OUT_DIR = RAW / "soilgrids"
BOUNDARY_PATH = RAW / "boundaries" / "study_districts.geojson"
BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["bdod", "cec", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"]
DEPTH = "0-5cm"
VALUE = "mean"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 20  # ISRIC's public API occasionally returns transient 503s
                         # under load; back off and retry rather than giving up.


def district_centroid(name: str):
    if BOUNDARY_PATH.exists():
        gdf = gpd.read_file(BOUNDARY_PATH)
        name_col = [c for c in gdf.columns if c.upper().startswith("NAME")][-1]
        match = gdf[gdf[name_col] == name]
        if len(match) == 1:
            c = match.geometry.iloc[0].centroid
            return c.y, c.x  # lat, lon
    lon_min, lat_min, lon_max, lat_max = DISTRICT_BBOXES[name]
    print(f"WARNING: using bbox-center fallback for {name} centroid -- run "
          f"01_get_boundaries.py for a real polygon centroid.")
    return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for district in DISTRICTS:
        lat, lon = district_centroid(district)
        params = {
            "lon": lon, "lat": lat,
            "property": PROPERTIES,
            "depth": DEPTH,
            "value": VALUE,
        }
        print(f"Querying SoilGrids for {district} (lat={lat:.4f}, lon={lon:.4f}) ...")
        data = None
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_exc = e
                print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"  waiting {RETRY_WAIT_SECONDS}s before retry...")
                    time.sleep(RETRY_WAIT_SECONDS)

        if data is None:
            print(f"  ERROR: all {MAX_RETRIES} attempts failed for {district}: {last_exc}")
            # Still record the district with known coordinates and null soil
            # values, rather than dropping the row entirely -- this keeps
            # the district present in the table (Integrity Class F/missing)
            # instead of silently vanishing from downstream joins.
            rows.append({"district": district, "lat": lat, "lon": lon,
                         **{p: None for p in PROPERTIES}})
            continue

        row = {"district": district, "lat": lat, "lon": lon}
        try:
            layers = data["properties"]["layers"]
            for layer in layers:
                prop_name = layer["name"]
                depth_entry = next(
                    d for d in layer["depths"] if d["label"] == DEPTH)
                raw_val = depth_entry["values"][VALUE]
                d_factor = layer["unit_measure"]["d_factor"]
                row[prop_name] = raw_val / d_factor if raw_val is not None else None
        except Exception as e:
            print(f"  WARNING: unexpected SoilGrids response shape for {district}: {e}")

        rows.append(row)
        time.sleep(1)  # be polite to the public API

    df = pd.DataFrame(rows)
    raw_json_note = OUT_DIR / "README_soilgrids_notes.txt"
    raw_json_note.write_text(
        "Values above are point queries at each district centroid, converted "
        "from SoilGrids' scaled integer units using each property's d_factor "
        "(documented at https://www.isric.org/explore/soilgrids/faq-soilgrids). "
        "For district-mean (rather than centroid) values, replace this point "
        "query with a WCS raster clip over the district polygon.\n"
    )

    out_csv = RAW.parent / "02_processed_data" / "soilgrids_district.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Always write a header row, even if every request failed, so downstream
    # scripts (and the QC step) get a well-formed empty table rather than a
    # zero-byte file that crashes pd.read_csv with EmptyDataError.
    if df.empty:
        df = pd.DataFrame(columns=["district", "lat", "lon"] + PROPERTIES)
    df.to_csv(out_csv, index=False)
    n_complete = df[PROPERTIES].notna().all(axis=1).sum() if not df.empty else 0
    print(f"Saved SoilGrids district table -> {out_csv} "
          f"({n_complete}/{len(DISTRICTS)} districts with complete data)")

    log_step(
        script="05_download_soilgrids.py", dataset="SoilGrids soil properties",
        provider="ISRIC", product_version="SoilGrids v2.0",
        resolution="250m (point query at centroid)", units="see data_dictionary.csv",
        query_parameters=f"properties={PROPERTIES}; depth={DEPTH}; value={VALUE}",
        output_path=str(out_csv), rows_or_files=len(df),
        status="OK" if n_complete == len(DISTRICTS) else "PARTIAL",
        notes=f"{n_complete}/{len(DISTRICTS)} districts complete; "
              f"point query at district centroid, not area-weighted mean; "
              f"ISRIC API returned transient 503s on some attempts" if n_complete < len(DISTRICTS) else
              "point query at district centroid, not area-weighted mean",
    )


if __name__ == "__main__":
    main()
