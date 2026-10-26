"""Build the PFT tables used in the paper: TRENDY v13 S2 landCoverFrac in the row layout of the main table.

For each model, writes <out_dir>/<MODEL>_nbp.feather with columns year, month, latitude, longitude,
PFT_0 .. PFT_{k-1} (float32, NaN where the source is NaN), one row per row of the main table with
year >= 2000, in the same order. Source: --landcover <pattern>, e.g. 'PFT/{}_S2_landCoverFrac.nc'
({} = model name), on the 0.5 deg grid; static (pft, lat, lon), annual or monthly (time, pft, lat, lon),
with time counted from 1700. Years after the last source year repeat the last source year (the same
calendar month for monthly files).

    python build_pft_feather.py --main data/data25_0616.feather --landcover 'PFT/{}_S2_landCoverFrac.nc' \
        --out_dir data/PFT_v13 JULES LPX-Bern SDGVM ...
"""
import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', required=True, help='main input table (Feather)')
    parser.add_argument('--landcover', required=True, help="landCoverFrac file pattern with {} for the model name")
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    keys = pf.read_table(args.main, columns=['year', 'month', 'latitude', 'longitude'], memory_map=True)
    yr = keys['year'].to_numpy()
    keys = keys.filter(pa.array(yr >= 2000))
    yr = keys['year'].to_numpy(); mo = keys['month'].to_numpy()
    la = keys['latitude'].to_numpy(); lo = keys['longitude'].to_numpy()
    print('rows', keys.num_rows, 'years', yr.min(), yr.max(), f'{time.time() - t0:.0f}s', flush=True)
    for m in args.models:
        ds = xr.open_dataset(args.landcover.format(m), decode_times=False)
        var = [k for k in ds.data_vars if ds[k].ndim >= 3][0]
        da = ds[var]
        if da.ndim == 3:
            kind, nt = 'static', 1
            arr = da.values.astype(np.float32)[None]
        else:
            nt = da.sizes[da.dims[0]]
            kind = 'monthly' if nt > 1000 else ('annual' if nt > 1 else 'static')
            arr = da.values.astype(np.float32)                       # (time, pft, lat, lon)
        assert arr.shape[-2:] == (360, 720), arr.shape
        if kind == 'annual':
            last = 1700 + nt - 1
            ti = np.minimum(yr, last) - 1700
        elif kind == 'monthly':
            last = 1700 + nt // 12 - 1
            ti = (np.minimum(yr, last) - 1700) * 12 + mo - 1
        else:
            last = None
            ti = np.zeros(len(yr), dtype=int)
        cols = {'year': keys['year'], 'month': keys['month'], 'latitude': keys['latitude'], 'longitude': keys['longitude']}
        for k in range(arr.shape[1]):
            cols[f'PFT_{k}'] = pa.array(arr[ti, k, la, lo], type=pa.float32())
        pf.write_feather(pa.table(cols), os.path.join(args.out_dir, f'{m}_nbp.feather'))
        print(m, kind, 'channels', arr.shape[1], 'last source year', last, f'{time.time() - t0:.0f}s', flush=True)
        del arr, cols


if __name__ == '__main__':
    main()
