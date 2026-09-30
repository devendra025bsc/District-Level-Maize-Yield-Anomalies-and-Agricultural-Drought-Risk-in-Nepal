# Nepal Maize Drought Q1 Reproducibility Project

This project rebuilds the Nepal maize-drought dataset (Jhapa, Ilam, Bhojpur, Morang,
Dhankuta, Sunsari; events 2015, 2016, 2018, 2022, 2024) from **authoritative,
acquisition-dated public sources**, replacing the single-snapshot Sentinel-1/2 arrays
and the placeholder (inter-annually constant) yield field identified in the prior
data-feasibility audit.

## Why this exists

The manuscript audit found two disqualifying problems in the original dataset:

1. **Sentinel-1/2 arrays were single, undated snapshots** (identical reflectance/backscatter
   range 500–2999 across all 30 district-years), so no temporal vegetation or radar
   response to drought could be computed.
2. **Maize yield was constant across years within each district**, so no yield
   prediction, correlation-with-time, or ML validation was defensible with the supplied
   values. A MoALD-consistent reconstruction was used provisionally in the manuscript,
   but this is not a substitute for verified, digitised district-year records.

This project does **not** patch around those problems. It re-pulls the underlying data
from source, with full provenance, so the manuscript can be upgraded honestly — or, if a
source turns out not to be recoverable, so the limitation can be documented precisely
instead of hidden.

## What I could not do for you automatically

I do not have credentials for the Copernicus Data Space Ecosystem, NASA Earthdata,
Google Earth Engine, or any MoALD/CBS portal login in this environment. **You need to
run the scripts in `07_code/` yourself**, after registering for free accounts and
placing your credentials where each script expects them (see each script's header).
Nothing here fabricates data — every script either pulls real data or fails loudly and
tells you what's missing.

The one dataset that has **no public API at all** is official MoALD/CBS district-year
maize yield. See `07_code/yield_moald_recovery_notes.md` for exactly what to look up and
how to enter it once you have it.

## Running from Jupyter / Colab instead of the command line

The scripts default to `python 07_code/NN_script.py` from the project root.
If you run them cell-by-cell in a notebook instead (e.g. paste the script
into a Jupyter cell, or `%run`), Python does not define `__file__` in that
context, so each script needs to know where `07_code/` lives another way.
Before running a script that way, either:

```python
%cd /path/to/Nepal_Maize_Drought_Q1/07_code
```

or set an environment variable once at the top of your notebook:

```python
import os
os.environ["NEPAL_DROUGHT_CODE_DIR"] = "/path/to/Nepal_Maize_Drought_Q1/07_code"
```

Every script checks `NEPAL_DROUGHT_CODE_DIR` first, then falls back to
`__file__`'s location, then to the current working directory -- so either
fix above resolves the `NameError: name '__file__' is not defined` you'll
otherwise hit on the `from project_config import ...` line.

## Run order

```
1. python 07_code/00_setup_check.py           # verifies credentials/config are present
2. python 07_code/01_get_boundaries.py         # Nepal district boundaries (GADM/HDX, no auth)
3. python 07_code/02_download_era5_land.py     # ERA5-Land daily climate (CDS API key)
4. python 07_code/03_download_chirps.py        # CHIRPS daily precip cross-check (no auth)
5. python 07_code/04_download_smap.py          # SMAP L4 soil moisture (NASA Earthdata login)
6. python 07_code/05_download_soilgrids.py     # ISRIC SoilGrids properties (no auth)
7. python 07_code/06_download_sentinel_gee.py  # acquisition-dated Sentinel-1/2 (GEE account)
8. python 07_code/07_compute_spi_spei.py       # SPI *and* SPEI from downloaded precip/PET
9. (manual) fill 01_raw_data/yield_moald/yield_moald_TEMPLATE.csv per the notes file
10. python 07_code/08_build_temporal_coverage_matrix.py
11. python 07_code/09_data_quality_checks.py   # QC plots -> 05_figures/qc/
```

Each script appends a row to `08_metadata/processing_log.csv` (provider, product,
version, resolution, units, download date, query parameters) as it runs, so every
number in the eventual manuscript stays traceable to code and a logged download.

## Directory map

| Folder | Contents |
|---|---|
| `00_manuscript/` | Manuscript drafts once analysis is finalised |
| `01_raw_data/` | Untouched downloads, one subfolder per source |
| `02_processed_data/` | District/event-aggregated, QC'd tables |
| `03_feature_engineering/` | Feature matrices for correlation/ML |
| `04_analysis/` | Statistics, LOEO-CV, ablation outputs |
| `05_figures/` | Regenerated figures (vector + raster) |
| `06_tables/` | Manuscript Tables 1–12 |
| `07_code/` | All download and processing scripts |
| `08_metadata/` | `data_dictionary.csv`, `source_urls.csv`, `processing_log.csv` |

## Data-integrity classification

Every value produced by this pipeline is tagged in `02_processed_data/*` with one of:

`A` verified observed data · `B` downloaded public data · `C` derived/computed ·
`D` reconstructed · `E` model output · `F` assumption · `G` interpretation ·
`H` literature-based

B–H are never relabelled as A.
