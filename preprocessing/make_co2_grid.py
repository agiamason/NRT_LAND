"""Monthly 1-degree field of atmospheric CO2 from the NOAA marine boundary layer (MBL) reference.

Input: the NOAA GML MBL surface reference file (co2_GHGreference.*_surface.txt, https://gml.noaa.gov/ccgg/mbl/),
weekly values in 41 latitude bands equally spaced in sine of latitude (value columns 1, 3, 5, ...). Weekly values
are grouped into months (5, 3, 4, 4, ... weeks per month; 5, 3, 4, 4, 4, 4, 4, 4, 4, 5, 3, 4 in leap years) and
averaged, interpolated linearly to 1-degree latitudes and repeated over longitude. In the last year of the file
(incomplete) all weeks are assigned to January, and the last month is dropped.

    python make_co2_grid.py co2_GHGreference.2020299850_surface.txt CO2.nc
"""
import calendar
import sys

import numpy as np
import pandas as pd
import xarray as xr


def main(path, out):
    with open(path) as f:
        rows = []
        for i, line in enumerate(f):
            if i == 0 or line.startswith('#'):
                continue
            rows.append(line.strip().split())

    co2 = pd.DataFrame(rows)
    test = co2.iloc[:, [0] + list(range(1, len(co2.columns), 2))]
    test_copy = test.copy()
    test_copy['year'] = test_copy[0].astype(float).astype(int)
    test_copy = test_copy.set_index('year')
    test_copy.index = pd.to_datetime(test_copy.index, format='%Y')
    test_copy.drop(0, axis=1, inplace=True)
    last_year = test_copy.index.max().year

    df = pd.DataFrame()
    for year in test_copy.index.unique():
        year_data = test_copy.loc[[year]]
        int_year = year.year
        if int_year == last_year:
            year_data['month'] = 1
            df_concat = year_data
        else:
            if calendar.isleap(int_year):
                group_sizes = [5, 3, 4, 4, 4, 4, 4, 4, 4, 5, 3, 4]
            else:
                group_sizes = [5, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            current_index = 0
            groups = []
            for label, size in enumerate(group_sizes, start=1):
                group = year_data.iloc[current_index:current_index + size].copy()
                group['month'] = np.repeat(label, len(group))
                current_index += size
                groups.append(group)
            df_concat = pd.concat(groups)
        df = pd.concat([df, df_concat])
    df.set_index('month', append=True, inplace=True)
    df = df.apply(pd.to_numeric)
    co2 = df.groupby([df.index.get_level_values(0), df.index.get_level_values(1)]).mean()

    sine_values = [-1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35,
                   -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                   0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]   # the 41 MBL latitude bands
    co2.columns = [np.arcsin(v) * (180 / np.pi) for v in sine_values]
    co2['time'] = pd.to_datetime(co2.index.get_level_values(0).year.astype(str) + '-'
                                 + co2.index.get_level_values(1).astype(str) + '-01', format='%Y-%m-%d')
    co2 = co2.reset_index(drop=True).set_index('time')

    ds = xr.Dataset()
    ds['CO2'] = xr.DataArray(co2.values, dims=['time', 'latitude'], coords={'time': co2.index, 'latitude': co2.columns})
    ds = ds.interp(latitude=np.arange(-89.5, 90, 1))
    ds = ds.expand_dims(dim={'longitude': np.arange(-179.5, 180, 1)}, axis=-1)
    ds = ds.transpose('time', 'latitude', 'longitude')
    ds['CO2'] = ds['CO2'].astype(float)
    ds = ds.isel(time=slice(0, -1))                                          # drop the incomplete last month
    ds.to_netcdf(out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
