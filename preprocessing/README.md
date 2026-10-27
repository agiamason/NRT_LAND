# Input tables

The emulator reads one main Feather table and, for models with land-cover input,
one PFT table per model (see "Data format" in the main README). This folder
downloads the inputs, puts them on the 0.5° grid (360 x 720) and builds the
tables.

| Step | Script | Output |
| --- | --- | --- |
| 1. TRENDY S2 model output (`nbp`, `landCoverFrac`, and `fFire`, `cVeg` for Supp Fig 10) | `download_trendy.sh` (TRENDY account needed) | one file per model and variable |
| 2. ERA5 monthly means: `u10`, `v10`, `t2m`, `sp`, `ssrd`, `strd`, `tp` and `q` at 1000 hPa | `download_era5_monthly.py` (CDS account needed) | 0.25° NetCDF |
| 3. Regridding with CDO: TRENDY NBP and land cover second-order conservative (`remapcon2,r720x360`), ERA5 bilinear (`remapbil,r720x360`, then `sellonlatbox,-180,180,-90,90`) | `regrid.sh` | `S2_nbp/`, `S2_PFT/`, `ERA5.nc`, `spfh.nc` |
| 4. TRENDY NBP of all models in one file, and the multi-model mean | `merge_trendy_nbp.py` | `Trendyv13_S2_nbp.nc`, `average_nbp.nc` |
| 4b. TRENDY land cover: coordinates, PFT dimension and time axis aligned | `prepare_landcover.py` | `PFT/<MODEL>_S2_landCoverFrac.nc` |
| 5. Atmospheric CO2: NOAA marine boundary layer reference (https://gml.noaa.gov/ccgg/mbl/), weekly, 41 latitude bands | `make_co2_grid.py` | `CO2.nc` (monthly, 1°) |
| 6. Main table | `build_input_table.py` | `<end_year>_<month>.feather` |
| 7. PFT tables | `build_pft_feather.py` | `<MODEL>_nbp.feather` |

```bash
bash preprocessing/download_trendy.sh USER /gcb2024/LAND/OUTPUT trendy_v13
python preprocessing/download_era5_monthly.py 2000 2024 era5
bash preprocessing/regrid.sh trendy_v13/raw trendy_v13 era5
python preprocessing/merge_trendy_nbp.py --in_dir trendy_v13/S2_nbp --start_year 2000 --end_year 2023 \
    --reccap_mask Land_reccap_regions_mask_05x05.nc --out Trendyv13_S2_nbp.nc --mean_out average_nbp.nc
python preprocessing/prepare_landcover.py --in_dir trendy_v13/S2_PFT --out_dir trendy_v13/PFT --end_year 2023
python preprocessing/make_co2_grid.py co2_GHGreference.2020299850_surface.txt CO2.nc
python preprocessing/build_input_table.py --start_year 2000 --end_year 2024 \
    --dgvm_path Trendyv13_S2_nbp.nc --dgvm_mean_path average_nbp.nc --feature_dir era5 \
    --pft_dir trendy_v13/PFT --co2_path CO2.nc --output_dir data
python preprocessing/build_pft_feather.py --main data/data25_0616.feather \
    --landcover 'trendy_v13/PFT/{}_S2_landCoverFrac.nc' --out_dir data/PFT_v13 \
    CABLE-POP CLASSIC IBIS iMAPLE ISBA-CTRIP JSBACH JULES LPX-Bern OCN ORCHIDEE SDGVM VISIT
```

Notes:

- In TRENDY v13, 19 models provide monthly S2 NBP (DLEM, LPJ-GUESS and LPJmL
  provide annual NBP only and are not used). Each keeps its last 288 months
  (2000-2023); CARDAMOM covers 2003-2023. The mean (`average_nbp`) excludes
  CARDAMOM.
- The land-cover files (`<MODEL>_S2_landCoverFrac.nc`, 12 models) are
  regridded like NBP and then aligned by `prepare_landcover.py`: annual files
  cover 1700-2023, monthly files 1700-01 to 2023-12; CABLE-POP, JSBACH and
  VISIT are static. `build_pft_feather.py` relies on this layout.
- All fields share one latitude order (south to north), so that flattening
  `(time, lat, lon)` gives the row order of the table.
- The paper's main table, `data25_0616.feather`, was built with an earlier
  notebook version of `build_input_table.py` from these files; its NBP and ERA5
  columns are identical to `Trendyv13_S2_nbp.nc`, `average_nbp.nc`, `ERA5.nc`
  and `spfh.nc`.
- `merge_trendy_nbp.py` and `make_co2_grid.py` reproduce the paper's
  `Trendyv13_S2_nbp.nc`, `average_nbp.nc` and `CO2.nc` exactly from the
  downloaded files. The ERA5 download requests were written for this
  repository from the metadata of the files used (monthly means, stream `moda`).
