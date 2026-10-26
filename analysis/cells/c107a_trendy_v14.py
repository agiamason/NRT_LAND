# TRENDY v13 (2010-2023) and v14 (2010-2024) S2 NBP for the 2024 test. LPJwsl is named LPJ-EOSIM in TRENDY v14;
# OCN and ISBA-CTRIP are not in v14.
_v13 = xr.open_dataset(TRENDY_V13_NBP).nbp.sel(time=slice('2010-01-01', '2023-12-31'))
_v14 = xr.open_dataset(TRENDY_V14_NBP).nbp.sel(time=slice('2010-01-01', '2024-12-31'))
assert _v13.time.size == 168 and _v14.time.size == 180
V14_NAME = {n: n for n in DGVM_list.values()}
V14_NAME.update({'visit-ut': 'VISIT-UT', 'LPJwsl': 'LPJ-EOSIM'})
_sec180 = sec_per_month(_v14).values
