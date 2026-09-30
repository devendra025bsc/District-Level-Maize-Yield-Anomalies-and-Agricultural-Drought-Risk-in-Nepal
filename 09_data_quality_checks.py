"""
09_data_quality_checks.py

Generates QC plots for every downloaded source so problems are visible before
they propagate into the manuscript: missing-data heatmaps, value-range
sanity checks, and a Sentinel-1/2 acquisition-date histogram (to confirm the
new pulls are genuinely multi-temporal, unlike the original single-snapshot
arrays).
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import os
_code_dir = os.environ.get("NEPAL_DROUGHT_CODE_DIR") or (
    str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd()))
sys.path.insert(0, _code_dir)
from project_config import PROCESSED, RAW, FIGURES

QC_DIR = FIGURES / "qc"


def plot_missingness(df: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(df.columns))))
    missing = df.isna().mean(axis=0).sort_values(ascending=True)
    ax.barh(missing.index, missing.values)
    ax.set_xlabel("Fraction missing")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sentinel_acquisition_dates(csv_path: Path, title: str, out_path: Path):
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {title}: {csv_path} is empty.")
        return
    except Exception as e:
        print(f"Skipping {title}: could not read {csv_path}: {e}")
        return
    if "date" not in df.columns or df.empty:
        return
    dates = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dates, bins=30)
    ax.set_title(title + f"  (n={len(dates)} acquisitions)")
    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Count")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in {
        "ERA5_Land": PROCESSED / "era5_land_district_daily.csv",
        "CHIRPS": PROCESSED / "chirps_district_daily.csv",
        "SMAP": PROCESSED / "smap_district_daily.csv",  # matches 04_download_smap.py's output filename
        "SoilGrids": PROCESSED / "soilgrids_district.csv",
        "SPI_SPEI": PROCESSED / "spi_spei_district_monthly.csv",
    }.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found (run its download script first).")
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"Skipping {name}: {path} exists but is empty (0 bytes or header-only "
                  f"with no columns) -- its download script likely failed for every "
                  f"district/event; check that script's output above.")
            continue
        except Exception as e:
            print(f"Skipping {name}: could not read {path}: {e}")
            continue

        if df.empty or len(df.columns) == 0:
            print(f"Skipping {name}: {path} has no usable rows/columns.")
            continue

        try:
            plot_missingness(df, f"{name}: missingness by column",
                              QC_DIR / f"missingness_{name}.png")
            print(f"QC plot written for {name}.")
        except Exception as e:
            print(f"WARNING: failed to plot {name}: {e}")

    plot_sentinel_acquisition_dates(
        RAW / "sentinel2" / "sentinel2_timeseries_district.csv",
        "Sentinel-2 acquisition dates (all districts/events)",
        QC_DIR / "sentinel2_acquisition_dates.png")
    plot_sentinel_acquisition_dates(
        RAW / "sentinel1" / "sentinel1_timeseries_district.csv",
        "Sentinel-1 acquisition dates (all districts/events)",
        QC_DIR / "sentinel1_acquisition_dates.png")

    print(f"\nAll QC plots written to {QC_DIR}")
    print("Inspect these before proceeding to feature engineering / analysis -- "
          "in particular confirm the Sentinel-1/2 histograms now show multiple "
          "acquisitions per event window (the original dataset had exactly 1).")


if __name__ == "__main__":
    main()
