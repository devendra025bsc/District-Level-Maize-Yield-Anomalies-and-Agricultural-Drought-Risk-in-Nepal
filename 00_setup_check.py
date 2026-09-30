"""
00_setup_check.py

Run this first. Verifies that credentials/config files required by later
scripts exist, and prints exactly what is missing and how to obtain it.
Does not download anything.
"""
import os
import sys
from pathlib import Path

CHECKS = []


def check(name, condition, fix_hint):
    CHECKS.append((name, bool(condition), fix_hint))


def main():
    home = Path.home()

    # CDS API (ERA5-Land)
    cdsapirc = home / ".cdsapirc"
    check(
        "Copernicus CDS API credentials (~/.cdsapirc)",
        cdsapirc.exists(),
        "Register at https://cds.climate.copernicus.eu, then create ~/.cdsapirc "
        "with your UID:API-key as documented at "
        "https://cds.climate.copernicus.eu/how-to-api",
    )

    # NASA Earthdata (SMAP via earthaccess)
    netrc = home / ".netrc"
    has_earthdata = False
    if netrc.exists():
        try:
            has_earthdata = "urs.earthdata.nasa.gov" in netrc.read_text()
        except Exception:
            has_earthdata = False
    check(
        "NASA Earthdata login (~/.netrc entry for urs.earthdata.nasa.gov)",
        has_earthdata,
        "Register at https://urs.earthdata.nasa.gov, then run "
        "`python -c \"import earthaccess; earthaccess.login(persist=True)\"` "
        "to write ~/.netrc automatically.",
    )

    # Google Earth Engine
    gee_creds = home / ".config" / "earthengine" / "credentials"
    check(
        "Google Earth Engine authentication",
        gee_creds.exists(),
        "Register a (free, non-commercial) GEE account at "
        "https://code.earthengine.google.com, then run "
        "`earthengine authenticate` (or `python -c \"import ee; ee.Authenticate()\"`).",
    )
    check(
        "EE_PROJECT environment variable (Cloud project for ee.Initialize)",
        bool(os.environ.get("EE_PROJECT")),
        "Create a Cloud project at https://console.cloud.google.com/projectcreate, "
        "register it for Earth Engine at "
        "https://code.earthengine.google.com/register, then set "
        "`os.environ['EE_PROJECT'] = 'your-project-id'` (notebook) or "
        "`export EE_PROJECT=your-project-id` (shell) before running "
        "06_download_sentinel_gee.py.",
    )

    # Python packages
    for pkg in ["cdsapi", "earthaccess", "ee", "geopandas", "xarray",
                "climate_indices", "sklearn"]:
        try:
            __import__(pkg)
            check(f"Python package: {pkg}", True, "")
        except ImportError:
            check(f"Python package: {pkg}", False,
                  f"pip install -r requirements.txt (missing: {pkg})")

    print("=" * 72)
    print("SETUP CHECK")
    print("=" * 72)
    all_ok = True
    for name, ok, hint in CHECKS:
        status = "OK " if ok else "MISSING"
        print(f"[{status}] {name}")
        if not ok:
            all_ok = False
            print(f"          -> {hint}")

    print("=" * 72)
    if all_ok:
        print("All checks passed. Proceed to 01_get_boundaries.py.")
    else:
        print("Some credentials/packages are missing (see above). Scripts that "
              "need them will refuse to run with fabricated data -- fix the "
              "items above first, or skip that specific source and note the "
              "gap in 08_metadata/processing_log.csv.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
