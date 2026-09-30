"""
07_compute_spi_spei.py

Computes SPI (Standardized Precipitation Index) AND SPEI (Standardized
Precipitation-Evapotranspiration Index) locally from the downloaded ERA5-Land
precipitation and temperature series, using the `climate_indices` package.
This directly fills Critical Gap C3 (SPEI absent from the original dataset,
Section 4.9 of the manuscript).

SPEI requires potential evapotranspiration (PET); this script estimates PET
via the Thornthwaite method from 2m temperature (adequate for a first pass --
swap in Penman-Monteith if wind/radiation/humidity are also pulled from
ERA5-Land for a more physically complete PET estimate).

References:
  McKee, Doesken & Kleist (1993) for SPI.
  Vicente-Serrano, Begueria & Lopez-Moreno (2010) for SPEI.

IMPORTANT: SPI/SPEI standardization requires a sufficiently long precipitation
climatology (ideally >=30 years) to fit the reference distribution. The five
event-window downloads in 02_download_era5_land.py are NOT sufficient on
their own -- this script expects a separate long-record climatology file
(see LONG_RECORD_NOTE below) and will refuse to compute indices without it,
rather than silently standardizing against an inadequate short record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import DISTRICTS, RAW, PROCESSED, log_step, require

LONG_RECORD_NOTE = """
SPI/SPEI STOP: no long-record precipitation climatology found.

This script looked for:
    01_raw_data/era5_land/era5_land_climatology_1991_2024.nc

The five short event-window downloads (Jan-Oct of single years) cannot be
used to fit a robust gamma/log-logistic reference distribution for
standardization -- doing so would produce misleading SPI/SPEI values dressed
up as rigorous indices.

To fix: extend 02_download_era5_land.py's request to pull a continuous
1991-2024 (or longer) daily precipitation + 2m temperature series for the
same district bounding boxes, save it as the filename above, then re-run this
script. This is a larger download (many years x 6 districts) so it is kept
as a separate, explicit step rather than folded into the default 5-event
pull.
"""


def main():
    long_record_path = RAW / "era5_land" / "era5_land_climatology_1991_2024.nc"
    if not long_record_path.exists():
        print(LONG_RECORD_NOTE)
        log_step(
            script="07_compute_spi_spei.py", dataset="SPI/SPEI", provider="computed locally",
            product_version="climate_indices", resolution="daily", units="unitless",
            query_parameters="-", output_path="-", rows_or_files=0,
            status="BLOCKED",
            notes="long-record climatology not available; see console message",
        )
        sys.exit(1)

    try:
        from climate_indices import indices, compute
    except ImportError:
        print("ERROR: climate_indices package not installed. "
              "pip install climate-indices")
        sys.exit(1)

    import xarray as xr
    ds = xr.open_dataset(long_record_path)

    results = []
    for district in DISTRICTS:
        # Expect the long-record file already aggregated to district-mean
        # daily series with a 'district' dimension/coordinate; adapt the
        # selection below to match however 02_download_era5_land.py's
        # extended pull structures its output.
        try:
            sub = ds.sel(district=district)
        except Exception as e:
            print(f"WARNING: could not select district {district} from "
                  f"climatology file ({e}); check the long-record file's schema.")
            continue

        precip_mm = sub["precip"].resample(time="1MS").sum().values  # monthly totals
        temp_c = sub["temp_2m"].resample(time="1MS").mean().values
        dates = sub["precip"].resample(time="1MS").sum()["time"].values

        # SPI (McKee et al. 1993), 3-month timescale as a starting point --
        # add 1/6/12-month timescales as needed for the manuscript's Section 2.10.
        spi_3 = indices.spi(
            precip_mm, scale=3, distribution=indices.Distribution.gamma,
            data_start_year=pd.Timestamp(dates[0]).year,
            calibration_year_initial=pd.Timestamp(dates[0]).year,
            calibration_year_final=pd.Timestamp(dates[-1]).year,
            periodicity=compute.Periodicity.monthly,
        )

        # PET via Thornthwaite, then SPEI (Vicente-Serrano et al. 2010)
        from climate_indices import palmer
        pet_mm = indices.pet(
            temperature_celsius=temp_c,
            latitude_degrees=sub.attrs.get("latitude", 27.0),
            data_start_year=pd.Timestamp(dates[0]).year,
        )
        water_balance = precip_mm - pet_mm
        spei_3 = indices.spei(
            precips_mm=precip_mm, pet_mm=pet_mm, scale=3,
            distribution=indices.Distribution.pearson,
            data_start_year=pd.Timestamp(dates[0]).year,
            calibration_year_initial=pd.Timestamp(dates[0]).year,
            calibration_year_final=pd.Timestamp(dates[-1]).year,
            periodicity=compute.Periodicity.monthly,
        )

        results.append(pd.DataFrame({
            "district": district, "date": dates,
            "SPI_3": spi_3, "SPEI_3": spei_3,
            "precip_mm": precip_mm, "PET_mm_thornthwaite": pet_mm,
        }))

    if not results:
        print("No districts processed -- check climatology file schema.")
        sys.exit(1)

    out = pd.concat(results, ignore_index=True)
    out_csv = PROCESSED / "spi_spei_district_monthly.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved SPI + SPEI district-monthly table -> {out_csv}")

    log_step(
        script="07_compute_spi_spei.py", dataset="SPI and SPEI",
        provider="computed locally from ERA5-Land", product_version="climate_indices",
        resolution="monthly, 3-month timescale", units="unitless (standardized)",
        query_parameters="scale=3; distribution=gamma(SPI)/pearson3(SPEI); PET=Thornthwaite",
        output_path=str(out_csv), rows_or_files=len(out), status="OK",
        notes="fills Critical Gap C3 (SPEI absent from original dataset)",
    )


if __name__ == "__main__":
    main()
