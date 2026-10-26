#!/bin/bash
# Put the inputs on the 0.5 deg grid of the emulator (360 x 720) with CDO, as for the paper's input files.
# Usage: regrid.sh TRENDY_RAW_DIR TRENDY_OUT_DIR ERA5_DIR
#   TRENDY_RAW_DIR: downloaded <MODEL>_S2_*nbp*.nc and <MODEL>_S2_*landCoverFrac.nc files (one per model);
#   regridded files go to TRENDY_OUT_DIR/S2_nbp and TRENDY_OUT_DIR/S2_PFT.
set -euo pipefail
RAW="${1:?TRENDY_RAW_DIR}"; OUT="${2:?TRENDY_OUT_DIR}"; ERA="${3:?ERA5_DIR}"

# TRENDY: second-order conservative remapping of the models that are not on the 0.5 deg grid; files already on it
# are copied. merge_trendy_nbp.py and prepare_landcover.py then align the coordinates (lat south to north,
# lon -179.75..179.75) and the time axis.
regrid_trendy() {   # file pattern, output directory
  mkdir -p "$2"
  for f in "$RAW"/$1; do
    if ncdump -h "$f" | grep -qE "(lon|longitude) = 720 ;"; then
      cp "$f" "$2/"
    else
      cdo remapcon2,r720x360 "$f" "$2/$(basename "$f")"
    fi
  done
}
regrid_trendy '*_S2*nbp*.nc' "$OUT/S2_nbp"
regrid_trendy '*_S2*landCoverFrac.nc' "$OUT/S2_PFT"

# ERA5 monthly means (download_era5_monthly.py): bilinear to 0.5 deg, longitudes -180..180
cdo -O --sortname mergetime "$ERA"/ERA5_single_levels_*.nc "$ERA/ERA5_025.nc"
cdo remapbil,r720x360 "$ERA/ERA5_025.nc" "$ERA/ERA5_720_360.nc"
cdo sellonlatbox,-180,180,-90,90 "$ERA/ERA5_720_360.nc" "$ERA/ERA5.nc"
cdo -O mergetime "$ERA"/ERA5_q1000_*.nc "$ERA/spfh_025.nc"
cdo remapbil,r720x360 "$ERA/spfh_025.nc" "$ERA/spfh_720_360.nc"
cdo sellonlatbox,-180,180,-90,90 "$ERA/spfh_720_360.nc" "$ERA/spfh.nc"
