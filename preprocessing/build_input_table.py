"""Build the emulator input tables (Feather) from gridded NetCDF inputs.

Main table <output_dir>/<end_year>_<last ERA5 month>.feather, one row per
(year, month, latitude index, longitude index) on the 0.5 deg grid, 2000 onwards:
  <MODEL>_nbp   TRENDY S2 monthly NBP of every model in --dgvm_path (DGVM dimension), kg m-2 s-1
  average_nbp   multi-model mean NBP (--dgvm_mean_path)
  u10, v10, t2m, sp, ssrd, strd, tp   ERA5 monthly means (--feature_dir/ERA5.nc)
  q             specific humidity (--feature_dir/spfh.nc)
  co2           marine boundary layer CO2 on a 1 deg grid (--co2_path: CO2.nc from make_co2_grid.py,
                or a Feather table with columns year, month and co2, rows year, month, lat, lon),
                repeated to 0.5 deg
PFT tables <output_dir>/PFT/<MODEL>_nbp.feather: TRENDY landCoverFrac of each model
(--pft_dir/<MODEL>_*.nc), static, annual (repeated for each month) or monthly; years
after the last year of the file repeat that year.

All NetCDF inputs must be on the same 0.5 deg grid (360 x 720) with the same latitude
order, so that flattening (time, lat, lon) matches the row order of the table.
The paper's table (data25_0616.feather: TRENDY v13 S2 NBP of the 19 models, ERA5 and
NOAA MBL CO2, 2000 onwards) was built with an earlier notebook version of this procedure.
The paper's PFT tables were built with build_pft_feather.py.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr

def get_start_date(s):
    pattern = r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?" 
    match = re.search(pattern, s)  
    if match:  
        year, month, day = match.groups() 
        return int(year), int(month), (int(day) if day else None)
    else:  
        print("No match found for date pattern.") 
        return None, None, None

def generate_empty_dataframe(start_year, end_year, latitude_len=360, longitude_len=720):
    years = range(start_year, end_year + 1)
    months = range(1, 13)
    lats = range(latitude_len)
    lons = range(longitude_len)
    
    idx = pd.MultiIndex.from_product([years, months, lats, lons],
                                    names=['year', 'month', 'latitude', 'longitude'])
    return idx.to_frame(index=False)

def extend_nrt(df_pft, base_year=2022, fill_years=[2023, 2024]):
    idx_base = df_pft[df_pft['year'] == base_year].index.values
    if len(idx_base) == 0:
        return df_pft
    
    pft_cols = [col for col in df_pft.columns if 'PFT' in col]
    for year in fill_years:
        idx_fill = df_pft[df_pft['year'] == year].index.values
        if len(idx_fill) > 0:
            df_pft.loc[idx_fill, pft_cols] = df_pft.loc[idx_base, pft_cols].values
    return df_pft

def run_processing(args):
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'PFT'), exist_ok=True)

    # 1. Generate Base Dataframe
    print(f"Generating dataframe for {args.start_year}-{args.end_year}...")
    df = generate_empty_dataframe(args.start_year, args.end_year)

    # 2. Read DGVM data
    print("Processing DGVM data...")
    dgvm_ds = xr.open_dataset(args.dgvm_path)
    dgvm_max_year = pd.to_datetime(dgvm_ds.time.values[-1]).year
    index_dgvm = df[df.year <= dgvm_max_year].index
    
    for idx, nbp_name in enumerate(dgvm_ds['DGVM']):
        col_name = f"{str(nbp_name.values)}_nbp"
        df.loc[index_dgvm, col_name] = dgvm_ds['nbp'][idx].values.flatten()

    dgvm_mean_ds = xr.open_dataset(args.dgvm_mean_path)
    df.loc[index_dgvm, f'average_nbp'] = dgvm_mean_ds['nbp'].values.flatten()

    # 3. Read ERA5 data
    print("Processing ERA5 data...")
    era5_ds = xr.open_dataset(os.path.join(args.feature_dir, 'ERA5.nc'))

    for var in list(era5_ds.coords):
        if 'time' in var:
            time_var = var
            break

    era5_time_last = pd.to_datetime(era5_ds[time_var].values[-1])

    era5_max_year, era5_max_month = era5_time_last.year, era5_time_last.month

    index_era5 = df[(df['year'] < era5_max_year) | 
                    ((df['year'] == era5_max_year) & (df['month'] <= era5_max_month))].index
    
    era5_cols = ['u10', 'v10', 't2m', 'sp', 'ssrd', 'strd', 'tp']
    for col in era5_cols:
        if col in era5_ds:
            df.loc[index_era5, col] = era5_ds[col].values.flatten()

    # 4. Read SPF H data
    spfh_ds = xr.open_dataset(os.path.join(args.feature_dir, 'spfh.nc'))
    if 'q' in spfh_ds:
        df.loc[index_era5, 'q'] = spfh_ds['q'].values.flatten()

    # 5. CO2 Data
    print("Processing CO2 data...")
    if args.co2_path.endswith('.nc'):
        co2_vals = xr.open_dataset(args.co2_path)['CO2'].sel(
            time=slice(f'{args.start_year}-01-01', f'{args.end_year}-12-31')).values
    else:
        ocean = pd.read_feather(args.co2_path)
        co2_vals = ocean[(ocean.year >= args.start_year) & (ocean.year <= args.end_year)]['co2'].values
    # Reshape and upscale (180x360 -> 360x720)
    co2_grid = co2_vals.reshape(-1, 180, 360)
    co2_upscaled = np.repeat(np.repeat(co2_grid, 2, axis=1), 2, axis=2)
    df['co2'] = co2_upscaled.flatten()

    # Save main feather file
    main_out = os.path.join(args.output_dir, f'{args.end_year}_{era5_max_month}.feather')
    df.to_feather(main_out)
    print(f"Saved main file to {main_out}")

    # 6. Process PFT Data
    print("Processing PFT directory...")
    for file_name in os.listdir(args.pft_dir):
        if not file_name.endswith('.nc'): continue
        
        print(f" Working on {file_name}")
        nbp_label = file_name.split('_')[0]
        df_pft = generate_empty_dataframe(args.start_year, args.end_year)
        pft_ds = xr.open_dataset(os.path.join(args.pft_dir, file_name))

        if 'time' in pft_ds.keys() and pft_ds.time.size > 1:
            delta_days = ((pft_ds.time[1] - pft_ds.time[0]) / np.timedelta64(1, 'D')).values
            
            t0 = pd.to_datetime(pft_ds.time[0].values)
            pft_start_year = t0.year
            pft_end_year = pd.to_datetime(pft_ds.time[-1].values).year
            
            # Case: Yearly data
            if 360 <= delta_days <= 366:
                start_omit = args.start_year - pft_start_year
                sel_idx = df_pft[df_pft['year'] <= pd.to_datetime(pft_ds.time[-1].values).year].index
                for c in range(pft_ds['landCoverFrac'].shape[1]):
                    arr = pft_ds['landCoverFrac'][start_omit:, c, ...].values
                    # Expand years to months
                    arr_repeated = np.repeat(arr, 12, axis=0)
                    df_pft.loc[sel_idx, f'PFT_{c}'] = arr_repeated.flatten()

            # Case: Monthly data
            elif 28 <= delta_days <= 31:

                start_omit = (args.start_year - pft_start_year) * 12

                # sel_idx = df_pft[df_pft.year <= args.end_year].index
                sel_idx = df_pft[df_pft.year <= pft_end_year].index
                
                # print(pft_start_year,  args.start_year, args.end_year)
                for c in range(pft_ds['landCoverFrac'].shape[1]):
                    arr = pft_ds['landCoverFrac'][start_omit:, c, ...].values
                    # print(df_pft[df_pft.year <= args.end_year].head(1))
                    # print(pft_ds.time[start_omit:])
                    
                    # print(arr.shape, sel_idx.shape)
                    df_pft.loc[sel_idx, f'PFT_{c}'] = arr.flatten()
            base_year= pft_end_year
            fill_years=[i for i in range(pft_end_year+1, args.end_year+1)]
            print(fill_years)
            df_pft = extend_nrt(df_pft, base_year=base_year, fill_years=fill_years)
        
        else:
            # Single time slice / Static data
            repeat_times = (args.end_year - args.start_year + 1) * 12
            data = np.squeeze(pft_ds['landCoverFrac'].values)
            for c in range(data.shape[0]):
                repeated = np.repeat(data[np.newaxis, c, ...], repeat_times, axis=0)
                df_pft[f'PFT_{c}'] = repeated.flatten()

        pft_out = os.path.join(args.output_dir, 'PFT', f'{nbp_label}_nbp.feather')
        df_pft.to_feather(pft_out)

def main():
    parser = argparse.ArgumentParser(description="Process climate and PFT data to Feather format.")
    
    # Time arguments
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2025)
    
    # Path arguments
    parser.add_argument("--dgvm_path", type=str, required=True, help='TRENDY S2 NBP of all models (dims DGVM, time, lat, lon)')
    parser.add_argument("--dgvm_mean_path", type=str, required=True, help='multi-model mean NBP (time, lat, lon)')
    parser.add_argument("--feature_dir", type=str, required=True, help='directory with ERA5.nc and spfh.nc')
    parser.add_argument("--pft_dir", type=str, required=True, help='directory with <MODEL>_*.nc landCoverFrac files')
    parser.add_argument("--co2_path", type=str, required=True, help='CO2.nc (make_co2_grid.py) or a Feather table with columns year, month and co2 on a 1 deg grid')
    parser.add_argument("--output_dir", type=str, default='./data')

    args = parser.parse_args()
    run_processing(args)

if __name__ == "__main__":
    main()

