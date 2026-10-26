"""Mask of the 36 cells of western Venezuela excluded from the evaluation.

In these cells the CRU-based forcing of the TRENDY models shows a decrease in
precipitation in 2021 that is absent from ERA5, and some DGVMs respond with
extreme fire emissions and carbon losses (Supp Fig 10). The cells are removed
from every evaluation (analysis/cells/c06_evaluation_mask.py).

Rule: 0.5 deg cells where CRU-JRA v2.4 precipitation (Rainf + Snowf) in 2021 is
below half of its 2010-2020 mean, and of these, the 8-connected region in
western Venezuela (it contains the cell centred at 8.25N, 71.75W).

    python make_event_mask.py OUT.nc
        writes the mask from event_mask_cells.txt (the cells used in the paper);

    python make_event_mask.py OUT.nc --crujra DIR
        recomputes the mask from the CRU-JRA v2.4 half-degree ORCHIDEE forcing files
        DIR/crujra_halfdeg_v2.4_<year>.nc (6-hourly, land points only) and checks it
        against event_mask_cells.txt.
"""
import argparse
import os

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = np.arange(-89.75, 90, 0.5)
LON = np.arange(-179.75, 180, 0.5)
SEED = (8.25, -71.75)


def cells_from_file():
    rows = [l.split() for l in open(os.path.join(HERE, 'event_mask_cells.txt')) if l.strip() and not l.startswith('#')]
    mask = np.zeros((LAT.size, LON.size), dtype=bool)
    for lat, lon in rows:
        mask[int(round((float(lat) - LAT[0]) / 0.5)), int(round((float(lon) - LON[0]) / 0.5))] = True
    return mask


def annual_precip(path):
    """Annual Rainf + Snowf (mm) of one CRU-JRA year on the global 0.5 deg grid (NaN off land)."""
    ds = xr.open_dataset(path, decode_times=False)
    ny, nx = ds.sizes['y'], ds.sizes['x']
    land = ds['land'].values.astype(np.int64) - 1          # 1-based index into the (y, x) grid
    yy, xx = np.divmod(land, nx)
    lat_pts = ds['nav_lat'].values[yy, xx]
    lon_pts = ds['nav_lon'].values[yy, xx]
    total = np.zeros(land.size)
    for var in ('Rainf', 'Snowf'):                           # kg m-2 s-1, 6-hourly steps
        total += ds[var].sum('tstep').values.astype(np.float64) * 21600.0
    ds.close()
    out = np.full((LAT.size, LON.size), np.nan)
    out[np.rint((lat_pts - LAT[0]) / 0.5).astype(int), np.rint((lon_pts - LON[0]) / 0.5).astype(int)] = total
    assert ny * nx == LAT.size * LON.size
    return out


def cells_from_crujra(directory):
    from scipy import ndimage
    path = os.path.join(directory, 'crujra_halfdeg_v2.4_{}.nc')
    base = np.mean([annual_precip(path.format(y)) for y in range(2010, 2021)], axis=0)
    p2021 = annual_precip(path.format(2021))
    ratio = p2021 / base
    low = np.isfinite(ratio) & (ratio < 0.5)
    labels, _ = ndimage.label(low, structure=np.ones((3, 3), dtype=int))
    seed = labels[int(round((SEED[0] - LAT[0]) / 0.5)), int(round((SEED[1] - LON[0]) / 0.5))]
    assert seed > 0, 'the seed cell is not below half of its 2010-2020 mean'
    return labels == seed, ratio, base


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--crujra', default='', help='directory of the CRU-JRA v2.4 half-degree files')
    args = parser.parse_args()

    mask = cells_from_file()
    data = {}
    if args.crujra:
        recomputed, ratio, base = cells_from_crujra(args.crujra)
        print(f'recomputed: {int(recomputed.sum())} cells; listed: {int(mask.sum())} cells; '
              f'identical: {bool((recomputed == mask).all())}')
        data['P2021_ratio'] = (('lat', 'lon'), ratio.astype(np.float32))
        data['P_2010_2020_mean'] = (('lat', 'lon'), base.astype(np.float32))
        mask = recomputed
    data['event_mask'] = (('lat', 'lon'), mask.astype(np.int8))
    ds = xr.Dataset(data, coords={'lat': LAT, 'lon': LON})
    ds['event_mask'].attrs['definition'] = ('1 = CRU-JRA v2.4 2021 precipitation < 0.5 x 2010-2020 mean, '
                                            '8-connected region in western Venezuela')
    ds.to_netcdf(args.out)
    print(f'wrote {args.out}: {int(mask.sum())} cells')


if __name__ == '__main__':
    main()
