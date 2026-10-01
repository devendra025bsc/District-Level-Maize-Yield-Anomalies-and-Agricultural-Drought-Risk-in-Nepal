# Nepal Drought Prediction — Plot Generation Code

Generates all 11 publication-quality manuscript figures from your uploaded data.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your data folder
Open `generate_all_plots.py` and change line 38:
```python
DATA_ROOT = r"C:\Users\YourName\Downloads\nepal_drought_data"
```
Point it to the folder containing all your uploaded files.

### 3. Run
```bash
# Generate all 11 figures
python generate_all_plots.py

# Generate specific figures only
python generate_all_plots.py --figs 1 2 6
python generate_all_plots.py --figs 11
```

---

## Expected Data Files in DATA_ROOT

| File | Used in |
|------|---------|
| `gadm41_NPL_3.json` | FIG01 |
| `study_districts.geojson` | FIG01, FIG05, FIG11 |
| `smap_district_daily.csv` | FIG02, FIG03, FIG05 |
| `sentinel2_timeseries_district.csv` | FIG04 |
| `chirps_*.csv` (13 files) | FIG02, FIG03, FIG05, FIG07–FIG10 |
| `era5_land_*.nc` (99 files) | FIG02, FIG03, FIG07–FIG10 |
| `SMAP_L4_SM_gph_*.h5` (9 files) | FIG11 |

---

## Output

All figures saved to `./outputs/figures/` as **PNG (300 dpi) + PDF + SVG**.

```
outputs/figures/
    FIG01/          FIG01_Study_Area.png/pdf/svg
    FIG02/          FIG02_2015_Drought_Verification.png/pdf/svg  × 5 events
    FIG03/          FIG03_Meteorological_2015.png/pdf/svg         × 5 events
    FIG04/          FIG04A_S2_Vegetation_2016.png/pdf/svg         × 4 events
                    FIG04B_S2_Index_Comparison.png/pdf/svg
    FIG05/          FIG05_Spatial_Drought_2015.png/pdf/svg        × 5 events
    FIG06/          FIG06_Yield_Analysis.png/pdf/svg
    FIG07_Ablation/ FIG07_Ablation_Study.png/pdf/svg
    FIG08_Feature_Importance/  FIG08_Feature_Importance.png/pdf/svg
    FIG09_Cross_Event/         FIG09_Cross_Event_Generalisation.png/pdf/svg
    FIG10_Early_Warning/       FIG10_Early_Warning.png/pdf/svg
    FIG11_SMAP/     FIG11A_SMAP_Temporal_Evolution.png/pdf/svg
                    FIG11B_SMAP_SM_Surface_Maps.png/pdf/svg
```

---

## Figure Descriptions

| Figure | Data Sources | Description |
|--------|-------------|-------------|
| FIG01 | GADM v4.1, study_districts.geojson | Study area: Nepal map, zoomed districts, maize area choropleth, production bars |
| FIG02 | ERA5-Land, CHIRPS, SMAP | Drought verification per event: precip anomaly, temp anomaly, SM anomaly, SPI |
| FIG03 | ERA5-Land, CHIRPS, SMAP | Meteorological evolution per event: 5 panels (precip, temp, ET, SM, SPI) |
| FIG04 | Sentinel-2 MSI | Vegetation index time series per event + multi-event box plots |
| FIG05 | SMAP, CHIRPS, Sentinel-2, MoALD | Spatial maps: SM anomaly, SPI, NDVI, yield anomaly |
| FIG06 | MoALD Nepal, CHIRPS, Sentinel-2 | Yield analysis: time series, yield–SPI scatter, yield–NDVI, heatmap |
| FIG07 | All features | Ablation study: R²/RMSE/MAE per feature group (LOEO-CV) |
| FIG08 | All features | RF feature importance + Pearson r + Spearman ρ |
| FIG09 | All features | Leave-one-event-out generalisation |
| FIG10 | ERA5-Land, CHIRPS, SMAP | Early-warning lead-time performance (1–8 weeks) |
| FIG11 | SMAP L4 HDF5 (NASA) | Soil moisture spatial maps + temporal evolution (2024 granules) |

---

## Notes

- **No titles or figure numbers** appear in any output — only panel labels (a)(b)(c)
- ERA5 files are monthly `.nc` files; the script automatically merges all months
- CHIRPS files are per-district-year CSVs; the script merges all automatically
- SMAP HDF5 files are full EASE-2 global grids; the script clips to Province 1
- Sentinel-2 input is a pre-compiled district-level timeseries CSV
- Yield data: the script uses MoALD Nepal values (real inter-annual variability) instead of the identical placeholder values in the source dataset
