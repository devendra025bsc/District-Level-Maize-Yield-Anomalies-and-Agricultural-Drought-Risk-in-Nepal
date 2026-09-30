"""
06_download_sentinel_gee.py

Downloads ACQUISITION-DATED Sentinel-2 (NDVI/EVI/NDWI/SAVI/NDRE) and
Sentinel-1 (VV/VH/VV-VH) district-mean time series for each event window,
via Google Earth Engine. This directly replaces the single undated snapshots
identified as Critical Problem C1/W1 in the prior data audit.

Requires a free GEE account -- see 00_setup_check.py.
References: Drusch et al. (2012) for Sentinel-2; Torres et al. (2012) for
Sentinel-1.

Cloud masking: Sentinel-2 SR_HARMONIZED collection's QA60/SCL band.
Terrain correction / speckle: Sentinel-1 GRD collection as delivered by GEE
already includes ellipsoid correction; for a rigorous terrain-flattened
gamma-naught product, use the COPERNICUS/S1_GRD_FLOAT collection with the
Vollrath et al. (2020) terrain-flattening approach documented at
https://developers.google.com/earth-engine/tutorials/community/sar-basics --
this script uses the standard GRD sigma-naught product and applies a simple
speckle-reduction (focal median) filter; swap in a full terrain-flattening
step if the manuscript requires it.
"""
import sys
from pathlib import Path

import ee
import pandas as pd
import geopandas as gpd

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, DISTRICT_BBOXES, EVENT_WINDOWS, RAW, log_step

OUT_DIR = RAW / "sentinel1"
OUT_DIR2 = RAW / "sentinel2"
BOUNDARY_PATH = RAW / "boundaries" / "study_districts.geojson"

CLOUD_FILTER_PCT = 40  # discard scenes with >40% cloud cover before per-pixel masking


def district_geometry_ee(name: str):
    if BOUNDARY_PATH.exists():
        gdf = gpd.read_file(BOUNDARY_PATH)
        name_col = [c for c in gdf.columns if c.upper().startswith("NAME")][-1]
        match = gdf[gdf[name_col] == name]
        if len(match) == 1:
            return ee.Geometry(match.geometry.iloc[0].__geo_interface__)
    lon_min, lat_min, lon_max, lat_max = DISTRICT_BBOXES[name]
    print(f"WARNING: using bbox fallback geometry for {name}.")
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def mask_s2_clouds(image):
    scl = image.select("SCL")
    # SCL classes 3=cloud shadow, 8/9=cloud medium/high prob, 10=cirrus, 11=snow
    mask = scl.remap([3, 8, 9, 10, 11], [0, 0, 0, 0, 0], 1)
    return image.updateMask(mask)


def add_s2_indices(image):
    nir = image.select("B8")
    red = image.select("B4")
    green = image.select("B3")
    blue = image.select("B2")
    re1 = image.select("B5")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename("EVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
    ndre = nir.subtract(re1).divide(nir.add(re1)).rename("NDRE")
    return image.addBands([ndvi, evi, ndwi, savi, ndre])


def pull_sentinel2(district, year, start, end, geom):
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER_PCT))
            .map(mask_s2_clouds)
            .map(add_s2_indices))

    n_images = coll.size().getInfo()
    if n_images == 0:
        return []

    def reduce_one(image):
        stats = image.select(["NDVI", "EVI", "NDWI", "SAVI", "NDRE"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=20, maxPixels=1e9)
        return ee.Feature(None, stats.set(
            "date", image.date().format("YYYY-MM-dd")).set(
            "cloud_pct", image.get("CLOUDY_PIXEL_PERCENTAGE")))

    feats = coll.map(reduce_one).getInfo()["features"]
    rows = []
    for feat in feats:
        p = feat["properties"]
        p.update({"district": district, "event_year": year})
        rows.append(p)
    return rows


def pull_sentinel1(district, year, start, end, geom):
    coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

    n_images = coll.size().getInfo()
    if n_images == 0:
        return []

    def reduce_one(image):
        vv = image.select("VV").focal_median(30, "circle", "meters")
        vh = image.select("VH").focal_median(30, "circle", "meters")
        vv_vh = vv.subtract(vh).rename("VV_VH")
        stack = vv.addBands(vh).addBands(vv_vh)
        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=20, maxPixels=1e9)
        return ee.Feature(None, stats.set(
            "date", image.date().format("YYYY-MM-dd")).set(
            "orbit", image.get("orbitProperties_pass")))

    feats = coll.map(reduce_one).getInfo()["features"]
    rows = []
    for feat in feats:
        p = feat["properties"]
        p.update({"district": district, "event_year": year})
        rows.append(p)
    return rows


def main():
    # Earth Engine now requires an explicit Cloud project for Initialize()
    # (registered for non-commercial EE use at
    # https://code.earthengine.google.com/register). Set it via the
    # EE_PROJECT environment variable, e.g.:
    #   export EE_PROJECT=your-project-id        (shell)
    #   os.environ["EE_PROJECT"] = "your-project-id"   (notebook, before this import)
    ee_project = os.environ.get("EE_PROJECT")
    if not ee_project:
        print("ERROR: no Earth Engine Cloud project set. Create one at "
              "https://console.cloud.google.com/projectcreate, register it at "
              "https://code.earthengine.google.com/register, then set it via "
              "the EE_PROJECT environment variable before running this script, e.g.\n"
              "    import os; os.environ['EE_PROJECT'] = 'your-project-id'")
        sys.exit(1)
    ee.Initialize(project=ee_project)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR2.mkdir(parents=True, exist_ok=True)

    s2_rows, s1_rows = [], []
    for district in DISTRICTS:
        geom = district_geometry_ee(district)
        for year, (start, end) in EVENT_WINDOWS.items():
            print(f"Pulling Sentinel-2 time series: {district} {year} ({start}:{end}) ...")
            s2_rows.extend(pull_sentinel2(district, year, start, end, geom))
            print(f"Pulling Sentinel-1 time series: {district} {year} ({start}:{end}) ...")
            s1_rows.extend(pull_sentinel1(district, year, start, end, geom))

    s2_df = pd.DataFrame(s2_rows)
    s1_df = pd.DataFrame(s1_rows)

    s2_csv = OUT_DIR2 / "sentinel2_timeseries_district.csv"
    s1_csv = OUT_DIR / "sentinel1_timeseries_district.csv"
    s2_df.to_csv(s2_csv, index=False)
    s1_df.to_csv(s1_csv, index=False)

    print(f"Saved acquisition-dated Sentinel-2 series ({len(s2_df)} scenes) -> {s2_csv}")
    print(f"Saved acquisition-dated Sentinel-1 series ({len(s1_df)} scenes) -> {s1_csv}")

    log_step(
        script="06_download_sentinel_gee.py", dataset="Sentinel-2 L2A (NDVI/EVI/NDWI/SAVI/NDRE)",
        provider="Copernicus/ESA via Google Earth Engine",
        product_version="COPERNICUS/S2_SR_HARMONIZED",
        resolution="10-20m per index, per-acquisition", units="unitless (indices)",
        query_parameters=f"cloud_filter<{CLOUD_FILTER_PCT}%; SCL cloud/shadow masked",
        output_path=str(s2_csv), rows_or_files=len(s2_df), status="OK",
        notes="acquisition-dated -- replaces single undated snapshot in original dataset",
    )
    log_step(
        script="06_download_sentinel_gee.py", dataset="Sentinel-1 GRD (VV/VH/VV-VH)",
        provider="Copernicus/ESA via Google Earth Engine",
        product_version="COPERNICUS/S1_GRD",
        resolution="10-20m, per-acquisition", units="dB",
        query_parameters="mode=IW; pol=VV+VH; focal_median 30m speckle filter",
        output_path=str(s1_csv), rows_or_files=len(s1_df), status="OK",
        notes="acquisition-dated -- replaces single undated snapshot in original dataset; "
              "standard sigma-naught GRD, not full terrain-flattened gamma-naught (see docstring)",
    )


if __name__ == "__main__":
    main()
