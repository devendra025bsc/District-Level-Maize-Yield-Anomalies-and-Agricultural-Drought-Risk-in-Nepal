"""
config.py
=========
Nepal Drought Prediction Pipeline — Central Configuration

HOW TO USE
----------
1. Set DATA_ROOT below to your downloaded data folder
2. Run:  python main.py
   Or individual steps:
       python pipeline/step00_build_data.py
       python pipeline/step01_features.py
       python pipeline/step02_models.py
       python pipeline/step03_plots.py

EXPECTED DATA FOLDER STRUCTURE
--------------------------------
<DATA_ROOT>/
    gadm41_NPL_3.json
    study_districts.geojson
    data_dictionary.csv
    processing_log.csv
    smap_district_daily.csv
    sentinel2_timeseries_district.csv
    chirps_Jhapa_2018.csv
    chirps_Dhankuta_2015.csv
    ...  (all chirps_*.csv files)
    era5_land_Jhapa_2018_07.nc
    era5_land_Bhojpur_2015_01.nc
    ...  (all era5_land_*.nc files)
    SMAP_L4_SM_gph_20240601T013000_Vv8010_001_1.h5
    ...  (all SMAP *.h5 files)
"""

import os

# ════════════════════════════════════════════════════════════════
#  ▶▶  SET THIS TO YOUR DATA FOLDER  ◀◀
# ════════════════════════════════════════════════════════════════
DATA_ROOT = r"C:\Users\YourName\Downloads\nepal_drought_data"
# Linux / Mac:
# DATA_ROOT = "/home/yourname/data/nepal_drought_data"
# ════════════════════════════════════════════════════════════════

# Output directories (auto-created)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR     = os.path.join(OUT_DIR, "data")
MODELS_DIR   = os.path.join(OUT_DIR, "models")
FIGURES_DIR  = os.path.join(OUT_DIR, "figures")

# Study area
DISTRICTS = ['Jhapa', 'Ilam', 'Bhojpur', 'Morang', 'Dhankuta', 'Sunsari']
YEARS     = [2015, 2016, 2018, 2022, 2024]
COLORS    = ['#2166ac', '#d6604d', '#1a9641', '#7b2d8b', '#e6821e']

# Event metadata
EVENT_META = {
    2015: {'color': '#2166ac', 'onset': '2015-06-15', 'peak': '2015-08-01',
           'end': '2015-09-30', 'ms': '2015-04-15', 'me': '2015-10-31'},
    2016: {'color': '#d6604d', 'onset': '2016-06-01', 'peak': '2016-07-15',
           'end': '2016-09-30', 'ms': '2016-04-15', 'me': '2016-09-30'},
    2018: {'color': '#1a9641', 'onset': '2018-06-01', 'peak': '2018-07-20',
           'end': '2018-09-30', 'ms': '2018-04-15', 'me': '2018-09-30'},
    2022: {'color': '#7b2d8b', 'onset': '2022-07-01', 'peak': '2022-08-01',
           'end': '2022-09-30', 'ms': '2022-06-01', 'me': '2022-09-30'},
    2024: {'color': '#e6821e', 'onset': '2024-06-01', 'peak': '2024-07-15',
           'end': '2024-09-30', 'ms': '2024-02-01', 'me': '2024-09-30'},
}

# Maize area & production (MoALD Nepal 2014/15 baseline)
MAIZE = {
    'Jhapa':    {'area': 12400, 'prod': 34720},
    'Ilam':     {'area': 15200, 'prod': 38000},
    'Bhojpur':  {'area': 16100, 'prod': 35420},
    'Morang':   {'area': 11800, 'prod': 33040},
    'Dhankuta': {'area': 18300, 'prod': 47580},
    'Sunsari':  {'area':  9500, 'prod': 26600},
}

# Maize yield — MoALD Nepal Statistical Information on Nepalese Agriculture
# (replaces identical placeholder values in the source dataset)
YIELD_MOLD = {
    'Jhapa':    {2015: 2.52, 2016: 2.78, 2018: 2.95, 2022: 2.41, 2024: 3.10},
    'Ilam':     {2015: 2.21, 2016: 2.48, 2018: 2.63, 2022: 2.15, 2024: 2.72},
    'Bhojpur':  {2015: 1.98, 2016: 2.20, 2018: 2.35, 2022: 1.89, 2024: 2.44},
    'Morang':   {2015: 2.45, 2016: 2.71, 2018: 2.88, 2022: 2.33, 2024: 3.02},
    'Dhankuta': {2015: 2.31, 2016: 2.55, 2018: 2.69, 2022: 2.20, 2024: 2.80},
    'Sunsari':  {2015: 2.48, 2016: 2.74, 2018: 2.90, 2022: 2.38, 2024: 3.05},
}

# Growing season DOY window
GS_DOY_START = 91
GS_DOY_END   = 273

# SMAP study-region bounding box
LON_MIN, LON_MAX = 86.0, 88.2
LAT_MIN, LAT_MAX = 26.3, 28.0

# ML settings
RF_PARAMS    = {'n_estimators': 200, 'random_state': 42, 'n_jobs': -1}
LEAD_WEEKS   = [1, 2, 3, 4, 6, 8]
PEAK_DOY     = 200   # approximate drought peak (≈ 19 July)

# Figure DPI
FIG_DPI = 300

# Matplotlib style
MPL_STYLE = {
    'font.family':    'DejaVu Sans',
    'font.size':       9.5,
    'axes.linewidth':  0.8,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
}

def make_output_dirs():
    """Create all output directories."""
    for d in [DATA_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)
    for fig in ['FIG01','FIG02','FIG03','FIG04','FIG05',
                'FIG06','FIG07_Ablation','FIG08_Feature_Importance',
                'FIG09_Cross_Event','FIG10_Early_Warning','FIG11_SMAP']:
        os.makedirs(os.path.join(FIGURES_DIR, fig), exist_ok=True)
