"""
Shared configuration for the Nepal Maize Drought Q1 reproducibility pipeline.

Edit DISTRICT_BBOXES only if 01_get_boundaries.py cannot resolve a district
polygon automatically (it will fall back to these approximate boxes).
Everything downstream should use the *polygon* (from boundaries), not the
bounding box, wherever a script needs district-mean values -- the boxes here
exist only as a fallback / sanity check.
"""

from __future__ import annotations
import csv
import datetime as dt
import os
from pathlib import Path


def this_dir() -> Path:
    """Directory containing the currently-running script.

    Falls back to the current working directory when `__file__` is not
    defined, which happens when a script is run cell-by-cell in Jupyter/
    ipykernel rather than via `python script.py`. If you're running in a
    notebook, `cd` into the `07_code/` folder first (or set the
    NEPAL_DROUGHT_CODE_DIR environment variable to its absolute path) so this
    resolves correctly.
    """
    env_override = os.environ.get("NEPAL_DROUGHT_CODE_DIR")
    if env_override:
        return Path(env_override).resolve()
    try:
        return Path(__file__).resolve().parent  # type: ignore[name-defined]
    except NameError:
        return Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = this_dir().parent
RAW = PROJECT_ROOT / "01_raw_data"
PROCESSED = PROJECT_ROOT / "02_processed_data"
FEATURES = PROJECT_ROOT / "03_feature_engineering"
ANALYSIS = PROJECT_ROOT / "04_analysis"
FIGURES = PROJECT_ROOT / "05_figures"
TABLES = PROJECT_ROOT / "06_tables"
METADATA = PROJECT_ROOT / "08_metadata"
PROCESSING_LOG = METADATA / "processing_log.csv"

# ---------------------------------------------------------------------------
# Study design (must match the audited manuscript's study scope)
# ---------------------------------------------------------------------------
DISTRICTS = ["Jhapa", "Ilam", "Bhojpur", "Morang", "Dhankuta", "Sunsari"]

# Operationally-defined event years and the ERA5-Land coverage windows
# documented in the manuscript audit (Table 1). These are the *provisional*
# windows used to scope each download; 07_compute_spi_spei.py and the event
# verification step may revise onset/peak/end after re-analysis.
EVENT_WINDOWS = {
    "2015": ("2015-01-01", "2015-10-31"),
    "2016": ("2016-01-01", "2016-09-30"),
    "2018": ("2018-01-01", "2018-09-30"),
    "2022": ("2022-06-01", "2022-09-30"),
    "2024": ("2024-02-01", "2024-09-30"),
}

# Approximate fallback bounding boxes (WGS84, [min_lon, min_lat, max_lon, max_lat]).
# Sourced from publicly known district extents for sanity-checking only --
# real district polygons should come from 01_get_boundaries.py (GADM/HDX).
DISTRICT_BBOXES = {
    "Jhapa":    [87.75, 26.35, 88.20, 26.85],
    "Ilam":     [87.75, 26.75, 88.25, 27.25],
    "Bhojpur":  [86.90, 27.00, 87.30, 27.45],
    "Morang":   [87.15, 26.35, 87.65, 26.85],
    "Dhankuta": [87.15, 26.75, 87.55, 27.15],
    "Sunsari":  [87.00, 26.40, 87.55, 26.75],
}

# Nepal maize growing-season prior (rainfed summer maize): used only as a
# starting search window for phenology fitting, NOT assumed as ground truth.
GROWING_SEASON_PRIOR = {"sow_start_month": 2, "sow_end_month": 5,
                         "harvest_start_month": 7, "harvest_end_month": 10}


def log_step(script: str, dataset: str, provider: str, product_version: str,
             resolution: str, units: str, query_parameters: str,
             output_path: str, rows_or_files, status: str, notes: str = "") -> None:
    """Append one row to 08_metadata/processing_log.csv. Call this at the end
    of every download/processing step so every number stays traceable."""
    METADATA.mkdir(parents=True, exist_ok=True)
    new_file = not PROCESSING_LOG.exists()
    with open(PROCESSING_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp_utc", "script", "dataset", "provider",
                              "product_version", "resolution", "units",
                              "query_parameters", "output_path",
                              "rows_or_files", "status", "notes"])
        writer.writerow([dt.datetime.utcnow().isoformat(timespec="seconds"),
                          script, dataset, provider, product_version, resolution,
                          units, query_parameters, output_path, rows_or_files,
                          status, notes])


def require(condition: bool, message: str) -> None:
    """Fail loudly instead of silently continuing with missing/fake data."""
    if not condition:
        raise RuntimeError(f"[DATA INTEGRITY STOP] {message}")
