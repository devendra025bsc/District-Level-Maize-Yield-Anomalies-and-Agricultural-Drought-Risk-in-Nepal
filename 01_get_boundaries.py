"""
01_get_boundaries.py

Downloads Nepal administrative boundaries (district / level-3) from GADM,
subsets to the six study districts, and saves a GeoJSON + shapefile to
01_raw_data/boundaries/. No authentication required.

Cross-check source: HDX COD-AB Nepal (https://data.humdata.org/dataset/cod-ab-npl)
is listed in 08_metadata/source_urls.csv -- download it manually and compare
district names/geometries if GADM naming does not match current Nepali
administrative terminology (post-2015 federal restructuring renamed/merged
some units; verify district names against Section 9 of the master prompt).
"""
import sys
import zipfile
import io
import requests
import geopandas as gpd

# Resolve the 07_code/ directory whether run as `python script.py` (where
# __file__ is defined) or cell-by-cell in Jupyter/ipykernel (where it isn't --
# in that case set NEPAL_DROUGHT_CODE_DIR or `cd` into 07_code/ first).
import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, RAW, log_step  # noqa: E402

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NPL_3.json.zip"
OUT_DIR = RAW / "boundaries"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GADM Nepal level-3 boundaries from {GADM_URL} ...")
    try:
        resp = requests.get(GADM_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: could not download GADM boundaries automatically: {e}")
        print("Manual fallback: download "
              "https://gadm.org/download_country.html (choose Nepal, level 3, "
              "GeoJSON) and place it at "
              f"{OUT_DIR / 'gadm41_NPL_3.json'}")
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(OUT_DIR)

    geojson_path = next(OUT_DIR.glob("*.json"))
    gdf = gpd.read_file(geojson_path)

    # GADM level-3 name column is typically NAME_3; adjust if GADM schema changes.
    name_col = "NAME_3" if "NAME_3" in gdf.columns else [
        c for c in gdf.columns if c.upper().startswith("NAME")][-1]

    subset = gdf[gdf[name_col].isin(DISTRICTS)].copy()
    found = sorted(subset[name_col].unique().tolist())
    missing = sorted(set(DISTRICTS) - set(found))

    if missing:
        print(f"WARNING: districts not matched by exact name in GADM: {missing}")
        print("Inspect the full district name list below and update DISTRICTS "
              "in project_config.py or add a name-mapping dict if GADM uses "
              "different spelling/administrative terminology:")
        print(sorted(gdf[name_col].unique().tolist()))

    out_geojson = OUT_DIR / "study_districts.geojson"
    subset.to_file(out_geojson, driver="GeoJSON")
    print(f"Saved {len(subset)} district polygons -> {out_geojson}")

    log_step(
        script="01_get_boundaries.py", dataset="Nepal district boundaries",
        provider="GADM", product_version="4.1", resolution="level-3 (district)",
        units="-", query_parameters=GADM_URL, output_path=str(out_geojson),
        rows_or_files=len(subset),
        status="OK" if not missing else "PARTIAL",
        notes=f"missing={missing}" if missing else "all 6 districts matched",
    )


if __name__ == "__main__":
    main()
