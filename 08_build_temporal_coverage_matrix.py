"""
08_build_temporal_coverage_matrix.py

Builds the temporal data-coverage matrix (Master Prompt Section 7, Table 3)
showing, for every district-event, which data sources are actually available
and at what temporal density -- e.g. "Sentinel-2: 4 cloud-free scenes" rather
than a binary yes/no. Run this after all download scripts above; it reads
only from 02_processed_data/ and 01_raw_data/yield_moald/, and never invents
a row for data that wasn't actually downloaded.
"""
import sys
from pathlib import Path

import pandas as pd

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, EVENT_WINDOWS, PROCESSED, RAW, TABLES, log_step


def count_rows(path: Path, district_col="district", year_col="event_year"):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if district_col not in df.columns or year_col not in df.columns:
        return {}
    return df.groupby([district_col, year_col]).size().to_dict()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    sources = {
        "ERA5_Land_days": PROCESSED / "era5_land_district_daily.csv",
        "CHIRPS_days": PROCESSED / "chirps_district_daily.csv",
        "SMAP_records": PROCESSED / "smap_district_raw.csv",
        "Sentinel2_scenes": RAW / "sentinel2" / "sentinel2_timeseries_district.csv",
        "Sentinel1_scenes": RAW / "sentinel1" / "sentinel1_timeseries_district.csv",
        "SPI_SPEI_months": PROCESSED / "spi_spei_district_monthly.csv",
    }

    counts = {name: count_rows(path) for name, path in sources.items()}

    yield_path = RAW / "yield_moald" / "yield_moald_TEMPLATE.csv"
    yield_status = {}
    if yield_path.exists():
        ydf = pd.read_csv(yield_path)
        for _, row in ydf.iterrows():
            yield_status[(row["district"], str(row["event_year"]))] = row["status"]

    rows = []
    for district in DISTRICTS:
        for year in EVENT_WINDOWS:
            row = {"district": district, "event_year": year}
            for name, count_dict in counts.items():
                row[name] = count_dict.get((district, year), 0)
            row["yield_status"] = yield_status.get((district, year), "TEMPLATE_NOT_FILLED")
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = TABLES / "Table3_temporal_data_coverage_matrix.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved temporal coverage matrix -> {out_csv}")
    print(df.to_string(index=False))

    n_missing_sources = (df.drop(columns=["district", "event_year", "yield_status"]) == 0).sum().sum()
    if n_missing_sources > 0:
        print(f"\nNOTE: {n_missing_sources} district-event x source cells have zero "
              "records. This is expected if you haven't run every download script "
              "yet, or if a source genuinely has no coverage for that window "
              "(e.g. SMAP predates 2015 for some periods) -- confirm which before "
              "treating a zero as a data gap worth reporting in the manuscript.")

    log_step(
        script="08_build_temporal_coverage_matrix.py", dataset="Temporal coverage matrix (Table 3)",
        provider="derived", product_version="-", resolution="district-event",
        units="record counts", query_parameters="-", output_path=str(out_csv),
        rows_or_files=len(df), status="OK", notes="",
    )


if __name__ == "__main__":
    main()
