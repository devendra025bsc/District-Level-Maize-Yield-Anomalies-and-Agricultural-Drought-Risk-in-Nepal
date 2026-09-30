"""
04_download_smap.py  (v3 -- fixes EASE-Grid 2.0 lat/lon extraction)

Downloads NASA SMAP L4 (SPL4SMGP, surface + root-zone soil moisture) for the
study districts/events via earthaccess, and aggregates to district-mean daily
values. Requires a free NASA Earthdata login -- see 00_setup_check.py.

WHAT CHANGED FROM v2:
  SPL4SMGP granules are NOT a simple lat/lon grid. They use the EASE-Grid 2.0
  projection with dimensions named 'y' and 'x', and latitude/longitude are
  stored as separate 2D coordinate arrays ('cell_lat', 'cell_lon') at the
  file root -- not as 1D 'lat'/'lon' dimensions. v2's `.sel(lat=..., lon=...)`
  therefore failed on every single granule ("'lat' is not a valid dimension").

  This version opens the root group for cell_lat/cell_lon, opens the
  Geophysical_Data group for sm_surface/sm_rootzone (same y/x grid, no
  coordinates attached), attaches the lat/lon as 2D non-dimension
  coordinates, and selects district pixels with a boolean mask instead of
  .sel() -- the correct approach for a projected (non-regular) grid.

Reference: Entekhabi et al. (2010), Proceedings of the IEEE 98(5):704-716.
"""
import sys
import datetime as dt
from pathlib import Path

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, DISTRICT_BBOXES, EVENT_WINDOWS, RAW, log_step

OUT_DIR = RAW / "smap"
SHORT_NAME = "SPL4SMGP"

# Which UTC hour(s) of each day to keep a granule for. [0] = midnight UTC
# only (1 granule/day, ~8x fewer downloads than the full 3-hourly product).
TARGET_HOURS_UTC = [0]


def granule_datetime(granule) -> "dt.datetime | None":
    try:
        umm = granule["umm"]
        time_range = umm["TemporalExtent"]["RangeDateTime"]
        start_str = time_range["BeginningDateTime"]
        return dt.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    except Exception:
        return None


def select_one_granule_per_day(granules, target_hours):
    from collections import defaultdict
    buckets = defaultdict(list)

    for g in granules:
        g_dt = granule_datetime(g)
        if g_dt is None:
            continue
        for target_hour in target_hours:
            hour_frac = g_dt.hour + g_dt.minute / 60.0
            diff = min(abs(hour_frac - target_hour), 24 - abs(hour_frac - target_hour))
            buckets[(g_dt.date(), target_hour)].append((diff, g, g_dt))

    selected = []
    for key, candidates in buckets.items():
        candidates.sort(key=lambda x: x[0])
        _, best_granule, best_dt = candidates[0]
        selected.append((best_granule, best_dt))
    return selected


def extract_district_means(filepath: str):
    """Open one SPL4SMGP HDF5 granule and return {district: {surface, rootzone}}
    by masking the EASE-Grid 2.0 x/y grid with each district's lat/lon bbox."""
    import h5py
    import numpy as np

    results = {}
    with h5py.File(filepath, "r") as f:
        # Root-level 2D coordinate arrays (EASE-Grid 2.0, dims [y, x])
        cell_lat = f["cell_lat"][:]
        cell_lon = f["cell_lon"][:]
        sm_surface = f["Geophysical_Data/sm_surface"][:]
        sm_rootzone = f["Geophysical_Data/sm_rootzone"][:]

        # SMAP fill value is typically -9999.0; mask it out before averaging
        fill_value = -9999.0
        sm_surface = np.where(np.isclose(sm_surface, fill_value), np.nan, sm_surface)
        sm_rootzone = np.where(np.isclose(sm_rootzone, fill_value), np.nan, sm_rootzone)

        for district in DISTRICTS:
            lon_min, lat_min, lon_max, lat_max = DISTRICT_BBOXES[district]
            mask = ((cell_lat >= lat_min) & (cell_lat <= lat_max) &
                    (cell_lon >= lon_min) & (cell_lon <= lon_max))

            if not mask.any():
                results[district] = {"surface": None, "rootzone": None}
                continue

            surf_vals = sm_surface[mask]
            root_vals = sm_rootzone[mask]
            surf_mean = float(np.nanmean(surf_vals)) if np.any(~np.isnan(surf_vals)) else None
            root_mean = float(np.nanmean(root_vals)) if np.any(~np.isnan(root_vals)) else None
            results[district] = {"surface": surf_mean, "rootzone": root_mean}

    return results


def main():
    import earthaccess
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auth = earthaccess.login(persist=True)
    if not auth.authenticated:
        print("ERROR: NASA Earthdata authentication failed. Run "
              "`python -c \"import earthaccess; earthaccess.login(persist=True)\"` "
              "and follow the prompts, or check ~/.netrc.")
        sys.exit(1)

    try:
        import h5py  # noqa: F401
    except ImportError:
        print("ERROR: h5py not installed. pip install h5py")
        sys.exit(1)

    min_lon = min(b[0] for b in DISTRICT_BBOXES.values())
    min_lat = min(b[1] for b in DISTRICT_BBOXES.values())
    max_lon = max(b[2] for b in DISTRICT_BBOXES.values())
    max_lat = max(b[3] for b in DISTRICT_BBOXES.values())

    all_rows = []
    for year, (start, end) in EVENT_WINDOWS.items():
        print(f"Searching SMAP {SHORT_NAME} granules for {year} ({start} to {end}) ...")
        results = earthaccess.search_data(
            short_name=SHORT_NAME,
            bounding_box=(min_lon, min_lat, max_lon, max_lat),
            temporal=(start, end),
        )
        print(f"  {len(results)} total 3-hourly granules found for this window.")
        if not results:
            print(f"  WARNING: no SMAP granules found for {year}.")
            continue

        selected = select_one_granule_per_day(results, TARGET_HOURS_UTC)
        print(f"  Reduced to {len(selected)} granules "
              f"(1 per day per target hour {TARGET_HOURS_UTC}) -- "
              f"downloading only these instead of all {len(results)}.")

        year_dir = OUT_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)

        n_extracted = 0
        for granule, g_dt in selected:
            try:
                files = earthaccess.download([granule], str(year_dir))
            except Exception as e:
                print(f"    WARNING: download failed for granule at {g_dt}: {e}")
                continue

            for f in files:
                try:
                    district_means = extract_district_means(f)
                    for district, vals in district_means.items():
                        if vals["surface"] is None and vals["rootzone"] is None:
                            continue
                        all_rows.append({
                            "district": district,
                            "event_year": year,
                            "date": g_dt.date().isoformat(),
                            "granule_time_utc": g_dt.isoformat(),
                            "file": Path(f).name,
                            "surface_soil_moisture": vals["surface"],
                            "rootzone_soil_moisture": vals["rootzone"],
                        })
                    n_extracted += 1
                except Exception as e:
                    print(f"    WARNING: could not extract districts from {f}: {e}")

        print(f"  Extracted {n_extracted}/{len(selected)} granules successfully for {year}.")

        log_step(
            script="04_download_smap.py", dataset="SMAP L4 soil moisture",
            provider="NASA NSIDC DAAC", product_version=f"{SHORT_NAME} v7",
            resolution=f"9km EASE-Grid 2.0, 1 granule/day at UTC hour(s) {TARGET_HOURS_UTC}",
            units="m3/m3",
            query_parameters=f"year={year}; window={start}:{end}; "
                              f"granules_available={len(results)}; granules_downloaded={len(selected)}",
            output_path=str(year_dir), rows_or_files=n_extracted,
            status="OK" if n_extracted else "NO DATA EXTRACTED",
            notes="Integrity Class F: one 3-hourly snapshot/day used as a daily proxy",
        )

    if all_rows:
        out_csv = RAW.parent / "02_processed_data" / "smap_district_daily.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\nSaved SMAP district-level table -> {out_csv} ({len(all_rows)} rows)")
        print("NOTE: each row is ONE 3-hourly snapshot used as that day's value "
              "(see 'granule_time_utc' column), not a true 24-hour average -- "
              "documented as Integrity Class F in processing_log.csv.")
    else:
        print("No SMAP data extracted -- see warnings above.")


if __name__ == "__main__":
    main()
