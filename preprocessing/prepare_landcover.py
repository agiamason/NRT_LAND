"""Align the TRENDY S2 land-cover fractions (landCoverFrac) for build_pft_feather.py.

Input: one file per model, already regridded to 0.5 deg (regrid.sh), named <MODEL>_S2_*landCoverFrac.nc. Output:
<out_dir>/<MODEL>_S2_landCoverFrac.nc with the variable landCoverFrac, dimensions (time,) PFT, lat, lon:
lat -89.75..89.75 (north-to-south files are flipped), lon -179.75..179.75 (files on 0..360 are shifted), the PFT
dimension renamed to PFT, and the time axis rebuilt to end in --end_year: annual files get one value per year
(1 January), monthly files one per month (15th); single-time and static files are left as they are. Models with
annual output only (DLEM, LPJ-GUESS, LPJmL) are skipped.

    python prepare_landcover.py --in_dir S2_PFT --out_dir PFT --end_year 2023
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

DGVMs_year = {'DLEM', 'LPJ-GUESS', 'LPJmL'}
PFT_DIM_NAMES = ['vegtype', 'pfts', 'pft', 'tile', 'type']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--end_year', type=int, default=2023)
    parser.add_argument('models', nargs='*', help='only these models (default: all files)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for file in sorted(os.listdir(args.in_dir)):
        if not file.endswith('landCoverFrac.nc'):
            continue
        dgvm_name = file.split('_')[0]
        if dgvm_name in DGVMs_year or (args.models and dgvm_name not in args.models):
            continue
        ds = xr.open_dataset(os.path.join(args.in_dir, file), decode_times=False, chunks={})   # lazy, low memory

        if 'latitude' in ds.coords:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in ds.coords:
            ds = ds.rename({'longitude': 'lon'})
        for dim_name in PFT_DIM_NAMES:
            if dim_name in ds['landCoverFrac'].dims:
                ds = ds.rename({dim_name: 'PFT'})
        if 'PFT' not in ds['landCoverFrac'].dims:
            # no PFT dimension: the PFTs are stored along the other dimension (CABLE-POP in TRENDY v13: 'time')
            other = [d for d in ds['landCoverFrac'].dims if d not in ('lat', 'lon')][0]
            ds = ds.rename({other: 'PFT'})
            if 'PFT' in ds.coords:
                ds = ds.drop_vars('PFT')

        if ds.lat.values[0] > ds.lat.values[-1]:
            ds = ds.reindex(lat=ds.lat[::-1])
        if ds.lon.values.max() > 180:
            ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby('lon')
            attrs = ds.lon.attrs.copy()
            ds = ds.assign_coords(lon=np.linspace(-179.75, 179.75, 720))
            ds.lon.attrs = attrs

        if 'time' in ds['landCoverFrac'].dims and ds.sizes['time'] > 1:
            n = ds.sizes['time']
            if n > 1000:      # monthly
                ds = ds.assign_coords(time=pd.date_range(end=f'{args.end_year}-12-01', periods=n, freq='MS')
                                      + pd.Timedelta(days=14))
            else:             # annual
                ds = ds.assign_coords(time=pd.date_range(end=f'{args.end_year}-01-01', periods=n, freq='YS'))

        ds = ds[['landCoverFrac']]
        out = os.path.join(args.out_dir, f'{dgvm_name}_S2_landCoverFrac.nc')
        ds.to_netcdf(out)
        print(dgvm_name, dict(ds['landCoverFrac'].sizes), flush=True)


if __name__ == '__main__':
    main()
