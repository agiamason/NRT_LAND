"""Merge the TRENDY S2 monthly NBP of all models into one file with a DGVM dimension, and write the multi-model mean.

Input: one file per model, already regridded to 0.5 deg (regrid.sh), named <MODEL>_S2_nbp.nc,
<MODEL>_S2_nbpAnnual.nc or <MODEL>_S2_Monthly_nbp.nc. Models with annual output only are skipped.
Coordinates are aligned to lat -89.75..89.75 (north-to-south files are flipped) and lon -179.75..179.75 (files on
0..360 are shifted to -180..180 and labelled -179.75..179.75, as for all 0.5 deg inputs). Each model keeps its last
(end_year - start_year + 1) x 12 months; a shorter record is placed at the end of the period
(CARDAMOM in TRENDY v13: 2003-2023). The mean (average_nbp.nc) excludes CARDAMOM and is set to NaN
outside the RECCAP2 land mask.

    python merge_trendy_nbp.py --in_dir S2_nbp --start_year 2000 --end_year 2023 \
        --reccap_mask Land_reccap_regions_mask_05x05.nc --out Trendyv13_S2_nbp.nc --mean_out average_nbp.nc
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

DGVMs_year = {'DLEM', 'LPJ-GUESS', 'LPJmL'}       # annual output only
PATTERN = r'^(.+?)_(S2_nbp\.nc|S2_nbpAnnual\.nc|S2_Monthly_nbp\.nc)$'


def mask_a(mask):
    """1 where `mask` has data, NaN elsewhere."""
    return xr.DataArray(np.where(np.isnan(mask), np.nan, 1), coords=mask.coords, dims=mask.dims)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--start_year', type=int, default=2000)
    parser.add_argument('--end_year', type=int, default=2023)
    parser.add_argument('--reccap_mask', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mean_out', required=True)
    args = parser.parse_args()
    n_months = (args.end_year - args.start_year + 1) * 12
    end = f'{args.end_year}-12-01'

    processed_datasets = []
    for root, dirs, files in os.walk(args.in_dir):
        for file in files:
            match = re.match(PATTERN, file)
            if not match:
                continue
            dgvm_name = match.group(1)
            if dgvm_name in DGVMs_year:
                continue
            ds = xr.open_dataset(os.path.join(root, file), decode_times=False)
            ds.coords['DGVM'] = dgvm_name

            if 'latitude' in ds.coords:
                ds = ds.rename({'latitude': 'lat'})
            if 'longitude' in ds.coords:
                ds = ds.rename({'longitude': 'lon'})
            if ds.lat.values[0] > ds.lat.values[-1]:
                ds = ds.reindex(lat=ds.lat[::-1])
            if ds.lon.values.max() > 180:
                ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby('lon')
                attrs = ds.lon.attrs.copy()
                ds = ds.assign_coords(lon=np.linspace(-179.75, 179.75, 720))
                ds.lon.attrs = attrs
            if 'nbp' in ds.data_vars:
                ds = ds[['nbp']]

            n = ds.sizes['time']
            if n < n_months:      # shorter record: the last n months of the period
                ds = ds.assign_coords(time=pd.date_range(end=end, periods=n, freq='MS') + pd.Timedelta(days=14))
            else:
                ds = ds.isel(time=slice(-n_months, None))
                ds = ds.assign_coords(time=pd.date_range(end=end, periods=n_months, freq='MS') + pd.Timedelta(days=14))
            processed_datasets.append(ds)

    final_dataset = xr.concat(processed_datasets, dim='DGVM')
    final_dataset.to_netcdf(args.out)
    print('models:', [str(x) for x in final_dataset.DGVM.values])

    land = xr.open_dataset(args.reccap_mask)
    land = land.reindex(latitude=land['latitude'][::-1])
    land = mask_a(land.mask).rename({'latitude': 'lat', 'longitude': 'lon'})
    mean_dataset = final_dataset.sel(DGVM=final_dataset.DGVM != 'CARDAMOM').mean(dim='DGVM')
    xr.Dataset({'nbp': mean_dataset.nbp * land}).to_netcdf(args.mean_out)


if __name__ == '__main__':
    main()
