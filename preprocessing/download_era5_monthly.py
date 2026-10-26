"""Download the ERA5 monthly means used as emulator inputs from the Copernicus Climate Data Store (cdsapi).

The input files of the paper hold ERA5 monthly averaged reanalysis (stream moda): u10, v10, t2m, sp (monthly means
of the analyses), ssrd, strd, tp (monthly means of the daily accumulations) on single levels, and specific
humidity q at 1000 hPa, on the native 0.25 deg grid; regrid.sh then puts them on the 0.5 deg grid. This script
writes the corresponding CDS requests (the original downloads were made interactively).

    python download_era5_monthly.py 2000 2024 OUT_DIR
"""
import os
import sys

import cdsapi

SINGLE = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'surface_pressure',
          'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'total_precipitation']


def main(y0, y1, out):
    os.makedirs(out, exist_ok=True)
    client = cdsapi.Client()
    years = [str(y) for y in range(int(y0), int(y1) + 1)]
    months = [f'{m:02d}' for m in range(1, 13)]
    client.retrieve('reanalysis-era5-single-levels-monthly-means', {
        'product_type': ['monthly_averaged_reanalysis'], 'variable': SINGLE, 'year': years, 'month': months,
        'time': ['00:00'], 'data_format': 'netcdf', 'download_format': 'unarchived'},
        os.path.join(out, f'ERA5_single_levels_{y0}_{y1}.nc'))
    client.retrieve('reanalysis-era5-pressure-levels-monthly-means', {
        'product_type': ['monthly_averaged_reanalysis'], 'variable': ['specific_humidity'], 'pressure_level': ['1000'],
        'year': years, 'month': months, 'time': ['00:00'], 'data_format': 'netcdf', 'download_format': 'unarchived'},
        os.path.join(out, f'ERA5_q1000_{y0}_{y1}.nc'))


if __name__ == '__main__':
    main(*sys.argv[1:4])
