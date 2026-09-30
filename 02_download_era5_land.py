"""
02_download_era5_land.py  (v5 -- fixes 'valid_time' vs 'time' coordinate rename)

Downloads ERA5-Land hourly reanalysis (2m temperature, total precipitation,
total evaporation) for each district bounding box and each event window,
then aggregates to district-mean daily values.

VERSION MARKER FOR AUTOMATED CHECKS: ERA5_LAND_SCRIPT_VERSION = "v5"

WHAT CHANGED FROM v4:
  Copernicus's current CDS backend returns NetCDF files where the time
  dimension/coordinate is named 'valid_time' instead of the older 'time'
  (visible in your traceback: "Variables on the dataset include ['t2m',
  'tp', 'e', 'number', 'valid_time', 'latitude', 'longitude', 'expver']").
  v4's aggregation step assumed 'time' and crashed with a KeyError. This
  version detects whichever name is present and renames it to 'time' before
  resampling, so it works with both old- and new-style CDS output.

  It also drops the extraneous 'number' and 'expver' dimensions some CDS
  responses include (ensemble/experiment-version bookkeeping fields that
  are irrelevant here and otherwise force awkward multi-index handling).

Still one district's small bounding box, one month, per API request (see
v4's docstring for why) -- that part is unchanged and working correctly.

Requires a free Copernicus CDS API key -- see 00_setup_check.py.
BEFORE FIRST USE: accept the dataset's license once at
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
(scroll down, click Accept), or every request fails regardless of API key.

Reference: Munoz-Sabater et al. (2021), Earth System Science Data 13(9):4349-4383.
"""
import sys
import time
import calendar
import traceback
from pathlib import Path

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import (DISTRICTS, DISTRICT_BBOXES, EVENT_WINDOWS, RAW,
                             log_step, require)

ERA5_LAND_SCRIPT_VERSION = "v5"

OUT_DIR = RAW / "era5_land"
MONTHLY_DIR = OUT_DIR / "monthly"
BOUNDARY_PATH = RAW / "boundaries" / "study_districts.geojson"

VARIABLES = ["2m_temperature", "total_precipitation", "total_evaporation"]
HOURS = ["00:00", "06:00", "12:00", "18:00"]

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 45
BETWEEN_REQUEST_PAUSE_SECONDS = 3


def district_bbox_padded(name: str, pad_deg: float = 0.05):
    try:
        import geopandas as gpd
        if BOUNDARY_PATH.exists():
            gdf = gpd.read_file(BOUNDARY_PATH)
            name_col = [c for c in gdf.columns if c.upper().startswith("NAME")][-1]
            match = gdf[gdf[name_col] == name]
            if len(match) == 1:
                lon_min, lat_min, lon_max, lat_max = match.geometry.iloc[0].bounds
                return (lon_min - pad_deg, lat_min - pad_deg,
                        lon_max + pad_deg, lat_max + pad_deg)
    except Exception:
        pass
    lon_min, lat_min, lon_max, lat_max = DISTRICT_BBOXES[name]
    return (lon_min - pad_deg, lat_min - pad_deg, lon_max + pad_deg, lat_max + pad_deg)


def month_chunks(start: str, end: str):
    import pandas as pd
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cur = start_ts.replace(day=1)
    while cur <= end_ts:
        year, month = cur.year, cur.month
        days_in_month = calendar.monthrange(year, month)[1]
        first_day = max(cur, start_ts).day
        last_day = min(cur.replace(day=days_in_month), end_ts).day
        yield year, month, first_day, last_day
        if month == 12:
            cur = cur.replace(year=year + 1, month=1)
        else:
            cur = cur.replace(month=month + 1)


def build_request(year, month, first_day, last_day, lon_min, lat_min, lon_max, lat_max,
                   use_new_schema: bool):
    req = {
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(first_day, last_day + 1)],
        "time": HOURS,
        "area": [lat_max, lon_min, lat_min, lon_max],
    }
    if use_new_schema:
        req["data_format"] = "netcdf"
        req["download_format"] = "unarchived"
    else:
        req["format"] = "netcdf"
    return req


def download_one_district_month(client, district, year, month, first_day, last_day,
                                 lon_min, lat_min, lon_max, lat_max):
    dist_dir = MONTHLY_DIR / district
    dist_dir.mkdir(parents=True, exist_ok=True)
    target = dist_dir / f"era5_land_{district}_{year}_{month:02d}.nc"

    if target.exists() and target.stat().st_size > 0:
        # Validate the cached file is actually readable before trusting it --
        # a prior interrupted download can leave a truncated/corrupt file.
        try:
            import xarray as xr
            with xr.open_dataset(target):
                pass
            print(f"      {target.name} already exists and is valid "
                  f"({target.stat().st_size/1e6:.2f} MB), skipping.")
            return target, "SKIPPED_EXISTING"
        except Exception as e:
            print(f"      {target.name} exists but failed to open ({e}) -- "
                  f"deleting and re-downloading.")
            target.unlink()

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        for use_new_schema in (True, False):
            request = build_request(year, month, first_day, last_day,
                                     lon_min, lat_min, lon_max, lat_max, use_new_schema)
            schema_label = "new" if use_new_schema else "legacy"
            print(f"      [{district} {year}-{month:02d}, attempt {attempt}/{MAX_RETRIES}, "
                  f"{schema_label} schema] requesting days {first_day}-{last_day}, "
                  f"{len(HOURS)} slots/day ...")
            try:
                client.retrieve("reanalysis-era5-land", request, str(target))
                print(f"      -> OK: {target.name} ({target.stat().st_size/1e6:.2f} MB)")
                time.sleep(BETWEEN_REQUEST_PAUSE_SECONDS)
                return target, "OK"
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                print(f"      !! FAILED ({schema_label}): {type(e).__name__}: {e}")
                if "licence" in msg or "license" in msg or "unauthorized" in msg or "401" in msg:
                    print("      >> LICENSE/AUTH error. Accept the license at "
                          "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land")
                    return None, f"FAILED (license/auth): {e}"
                if "cost limit" in msg or "too large" in msg or "403" in msg:
                    print("      >> Try reducing HOURS to 2 slots/day near the top of "
                          "this script.")
                    return None, f"FAILED (cost limit): {e}"
        if attempt < MAX_RETRIES:
            print(f"      Waiting {RETRY_WAIT_SECONDS}s before retry...")
            time.sleep(RETRY_WAIT_SECONDS)

    print(f"      ALL ATTEMPTS FAILED for {district} {year}-{month:02d}.")
    traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
    return None, f"FAILED: {type(last_exc).__name__}: {last_exc}"


def _normalize_time_coord(ds):
    """CDS's current backend names the time coordinate 'valid_time'; older
    responses use 'time'. Normalize to 'time' either way, and drop the
    'number' (ensemble member) and 'expver' (experiment version) dimensions
    that are irrelevant for a deterministic reanalysis single-run pull."""
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    for extra_dim in ("number", "expver"):
        if extra_dim in ds.dims:
            ds = ds.isel({extra_dim: 0}, drop=True)
        elif extra_dim in ds.coords:
            ds = ds.drop_vars(extra_dim)
    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError(
            f"No usable time coordinate found after normalization. "
            f"Available variables: {list(ds.variables)}")
    return ds


def aggregate_district_month_to_daily(nc_path: Path, district: str, year: str):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    ds = _normalize_time_coord(ds)
    daily = ds.resample(time="1D").mean()
    df = daily.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
    df["district"] = district
    df["event_year"] = year
    return df


def main():
    import cdsapi
    import pandas as pd

    print(f"ERA5-Land download script version: {ERA5_LAND_SCRIPT_VERSION}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = cdsapi.Client()
    except Exception as e:
        print(f"ERROR: could not create CDS client -- check ~/.cdsapirc: {e}")
        sys.exit(1)

    all_daily = []
    n_ok, n_failed, n_skipped = 0, 0, 0

    for district in DISTRICTS:
        lon_min, lat_min, lon_max, lat_max = district_bbox_padded(district)
        print("=" * 72)
        print(f"DISTRICT: {district}  bbox=({lon_min:.3f},{lat_min:.3f},"
              f"{lon_max:.3f},{lat_max:.3f})")
        print("=" * 72)

        for year, (start, end) in EVENT_WINDOWS.items():
            print(f"  EVENT YEAR {year} ({start} to {end})")
            for y, m, first_day, last_day in month_chunks(start, end):
                nc_path, status = download_one_district_month(
                    client, district, y, m, first_day, last_day,
                    lon_min, lat_min, lon_max, lat_max)

                if nc_path is None:
                    n_failed += 1
                    log_step(
                        script="02_download_era5_land.py", dataset="ERA5-Land",
                        provider="Copernicus Climate Data Store (ECMWF)",
                        product_version=f"reanalysis-era5-land ({ERA5_LAND_SCRIPT_VERSION})",
                        resolution=f"~9km, {len(HOURS)} slots/day, per-district/month",
                        units="K->C (temp), m->mm/day (precip, evap)",
                        query_parameters=f"district={district}; {y}-{m:02d}",
                        output_path="-", rows_or_files=0, status="FAILED", notes=status,
                    )
                    continue

                if status == "OK":
                    n_ok += 1
                else:
                    n_skipped += 1

                try:
                    daily = aggregate_district_month_to_daily(nc_path, district, year)
                    all_daily.append(daily)
                except Exception as e:
                    n_failed += 1
                    print(f"      WARNING: downloaded but failed to aggregate "
                          f"{district} {y}-{m:02d}: {e}")
                    log_step(
                        script="02_download_era5_land.py", dataset="ERA5-Land",
                        provider="Copernicus Climate Data Store (ECMWF)",
                        product_version=f"reanalysis-era5-land ({ERA5_LAND_SCRIPT_VERSION})",
                        resolution="-", units="-",
                        query_parameters=f"district={district}; {y}-{m:02d}",
                        output_path=str(nc_path), rows_or_files=0,
                        status="FAILED_AGGREGATION", notes=str(e),
                    )

        log_step(
            script="02_download_era5_land.py", dataset="ERA5-Land",
            provider="Copernicus Climate Data Store (ECMWF)",
            product_version=f"reanalysis-era5-land ({ERA5_LAND_SCRIPT_VERSION})",
            resolution=f"~9km, {len(HOURS)} slots/day->daily, per-district/month",
            units="K->C (temp), m->mm/day (precip, evap)",
            query_parameters=f"district={district}; all 5 event windows",
            output_path=str(MONTHLY_DIR / district), rows_or_files="see console",
            status="OK", notes="",
        )

    print("\n" + "=" * 72)
    print(f"DOWNLOAD SUMMARY: {n_ok} new, {n_skipped} cached, {n_failed} failed "
          f"(out of {n_ok + n_skipped + n_failed} district-month requests)")
    print("=" * 72)

    if not all_daily:
        print("No data downloaded successfully. See errors above.")
        sys.exit(1)

    out_csv = RAW.parent / "02_processed_data" / "era5_land_district_daily.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_daily, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved district-daily ERA5-Land table -> {out_csv}  ({len(combined)} rows)")
    if n_failed > 0:
        print(f"NOTE: {n_failed} district-month pieces failed and are NOT included -- "
              f"see 08_metadata/processing_log.csv for which ones, and re-run this "
              f"script to retry only the missing pieces.")
    require(out_csv.exists(), "ERA5-Land aggregation failed to write output.")


if __name__ == "__main__":
    main()
