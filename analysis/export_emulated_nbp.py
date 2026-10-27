"""Write the emulated monthly NBP of the paper as NetCDF files, with the evaluation mask.

    python analysis/export_emulated_nbp.py OUT_DIR_OF_PACKAGE

Reads the 5-fold mean emulator output (EMULATOR_EXP1_DIR, EMULATOR_EXP2_DIR, see config.py) and the evaluation mask
written by the analysis (OUT_DIR/valid_mask_final.nc). Writes:
  emulated_nbp_2022-2023.nc  experiment 1: emulators trained on 2000-2021, run recursively from January 2022
  emulated_nbp_2024.nc       experiment 2: emulators trained on 2000-2023, run recursively from January 2024
  evaluation_mask.nc         grid cells used in the evaluation of each model
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

MODELS = {101: 'ORCHIDEE', 102: 'CABLE-POP', 103: 'CLASSIC', 104: 'IBIS', 105: 'JSBACH', 106: 'JULES', 107: 'LPX-Bern',
          108: 'OCN', 109: 'VISIT', 110: 'iMAPLE', 111: 'ISBA-CTRIP', 112: 'CLM6.0', 113: 'LPJwsl', 114: 'EDv3',
          115: 'ISAM', 116: 'E3SM', 117: 'CARDAMOM', 118: 'SDGVM', 119: 'VISIT-UT'}
LAT = np.arange(-89.75, 90, 0.5)
LON = np.arange(-179.75, 180, 0.5)
NBP_ATTRS = {'long_name': 'Net biome production simulated by the emulator', 'units': 'kg m-2 s-1',
             'standard_name': 'surface_net_downward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_all_land_processes',
             'comment': 'Positive values are a net carbon uptake by the land. Mean of the five networks of each emulator. '
                        'NaN where the target model has no output'}


def read_months(path, years):
    t = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(include_columns=['year', 'month', 'lat', 'lon', 'predictions']))
    y = t.column('year').to_numpy().astype(int)
    keep = np.isin(y, years)
    m = t.column('month').to_numpy().astype(int)[keep]
    ti = (y[keep] - years[0]) * 12 + m - 1
    a = np.full((len(years) * 12, LAT.size, LON.size), np.nan, dtype=np.float32)
    a[ti, t.column('lat').to_numpy()[keep].astype(int), t.column('lon').to_numpy()[keep].astype(int)] = \
        t.column('predictions').to_numpy(zero_copy_only=False)[keep] / 1e10
    assert int(keep.sum()) == len(years) * 12 * LAT.size * LON.size, (path, int(keep.sum()))
    return a


def write(out, years, src_dir, title, summary, mask_note=' in that month.'):
    arr = np.stack([read_months(os.path.join(src_dir, f'{k}.csv'), years) for k in MODELS])
    time = pd.date_range(f'{years[0]}-01-01', periods=len(years) * 12, freq='MS') + pd.Timedelta(days=14)
    attrs = dict(NBP_ATTRS, comment=NBP_ATTRS['comment'] + mask_note)
    ds = xr.Dataset({'nbp': (('model', 'time', 'lat', 'lon'), arr, attrs)},
                    coords={'model': list(MODELS.values()), 'time': time, 'lat': LAT, 'lon': LON})
    ds['lat'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
    ds['lon'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
    ds['model'].attrs = {'long_name': 'target model (TRENDY v13, S2 simulation)'}
    ds.attrs = {'title': title, 'summary': summary, 'Conventions': 'CF-1.8', 'license': 'CC BY 4.0'}
    enc = {'nbp': {'zlib': True, 'complevel': 4, 'chunksizes': (1, 1, LAT.size, LON.size), '_FillValue': np.float32(np.nan)}}
    ds.to_netcdf(out, encoding=enc)
    print(out, dict(ds.sizes), f'{os.path.getsize(out) / 1e6:.0f} MB')


def main(dst):
    os.makedirs(dst, exist_ok=True)
    common = ('Monthly net biome production (NBP) on a 0.5 degree grid predicted by convolutional neural network emulators '
              'of 19 models of the TRENDY v13 ensemble (S2 simulation). The emulators are run with ERA5 climate, atmospheric '
              'CO2 and, for 12 models, PFT fractions, and predict each month from the inputs of that month and the preceding '
              '12 months, using their own predictions as NBP history from the first predicted month onwards. ')
    write(os.path.join(dst, 'emulated_nbp_2022-2023.nc'), [2022, 2023], config.EMULATOR_EXP1_DIR,
          'Emulated monthly NBP, 2022-2023',
          common + 'Emulators trained on 2000-2021; predictions from January 2022 to December 2023.')
    write(os.path.join(dst, 'emulated_nbp_2024.nc'), [2024], config.EMULATOR_EXP2_DIR,
          'Emulated monthly NBP, 2024',
          common + 'Emulators trained on 2000-2023; predictions from January to December 2024. Run with the '
                   'PFT fractions of 2023.', mask_note=' in the same month of 2023.')

    vm = xr.open_dataset(config.OUT_DIR + 'valid_mask_final.nc')
    names = ['VISIT-UT' if str(n) == 'visit-ut' else str(n) for n in vm.DGVM.values]
    mask = xr.Dataset(
        {'evaluation_mask': (('model', 'lat', 'lon'), vm['valid_mask_final'].values.astype(np.int8),
                             {'long_name': 'grid cells used to evaluate the emulator of each target model',
                              'comment': '1 = valid_mask = 1 and event_mask = 0'}),
         'valid_mask': (('model', 'lat', 'lon'), vm['valid_mask'].values.astype(np.int8),
                        {'long_name': 'target-model NBP complete over 2010-2023 with temporal standard deviation above 1e-10 kgC m-2 s-1'}),
         'event_mask': (('lat', 'lon'), vm['event_mask'].values.astype(np.int8),
                        {'long_name': '36 cells of western Venezuela excluded from the evaluation',
                         'comment': 'cells where the CRU-based forcing shows a decrease in precipitation in 2021 that is absent from ERA5; selected where CRU-JRA v2.4 '
                                    'precipitation in 2021 is below half of its 2010-2020 mean (8-connected region)'})},
        coords={'model': names, 'lat': LAT, 'lon': LON})
    mask.attrs = {'title': 'Evaluation mask of the emulated NBP', 'Conventions': 'CF-1.8', 'license': 'CC BY 4.0'}
    out = os.path.join(dst, 'evaluation_mask.nc')
    mask.to_netcdf(out, encoding={v: {'zlib': True, 'complevel': 4} for v in mask.data_vars})
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
