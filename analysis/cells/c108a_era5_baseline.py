# ERA5 regression baseline: 8 predictors + intercept per month (tropical and extratropical land ERA5 t2m and tp
# anomalies, lag 0 and mean of the 3 previous months); helpers own_anom, _ss (skill score), _rmse.
def era5_predictors(t_end='2024-12-31'):
    """8 predictors + intercept, one row per month from 2010-01: ERA5 t2m and tp land-area-weighted anomalies (vs 2010-2021 monthly
    climatology) for the tropics (|lat| < 23.5) and extratropics, at lag 0 and as the mean of t-3..t-1. Land = RECCAP mask cells (cell 16).
    ERA5 longitudes (-180..179.5) are aligned to the model grid (-179.75..179.75) by index."""
    ds = xr.open_dataset(ERA5_FILE)
    tt = ds['valid_time'].values
    idx = np.where((tt >= np.datetime64('2009-10-01')) & (tt <= np.datetime64(t_end)))[0]
    t2 = ds['t2m'].isel(valid_time=idx).values.astype(np.float64)
    tp = ds['tp'].isel(valid_time=idx).values.astype(np.float64)
    ds.close()
    lat = model_results['ORCHIDEE'].lat.values; lon = model_results['ORCHIDEE'].lon.values
    area = seaflux.area.area_grid(lat, lon, return_dataarray=True).values
    land = np.isfinite(R_720_360.values)
    trop = (np.abs(lat) < 23.5)[:, None] & np.ones((1, lon.size), dtype=bool)
    nt = t2.shape[0]; yrs = 2009 + (np.arange(nt) + 9) // 12; mos = (np.arange(nt) + 9) % 12 + 1
    cols = {}
    for k, m in {'trop': land & trop, 'extratrop': land & ~trop}.items():
        for vn, v in [('T', t2), ('P', tp)]:
            s = (v[:, m] * area[m]).sum(1) / area[m].sum()
            clim = np.array([s[(yrs >= 2010) & (yrs <= 2021) & (mos == mm)].mean() for mm in range(1, 13)])
            a = s - clim[mos - 1]
            prev3 = np.r_[np.full(3, np.nan), [a[i - 3:i].mean() for i in range(3, nt)]]
            cols[f'{vn}_{k}_lag0'] = a[3:]; cols[f'{vn}_{k}_prev3'] = prev3[3:]
    return np.column_stack([np.ones(nt - 3)] + [cols[k] for k in sorted(cols)])

X_ERA5 = era5_predictors()            # 180 rows: 2010-01 .. 2024-12
def own_anom(s):                      # monthly series starting 2010-01, minus its own 2010-2021 monthly climatology
    s = np.asarray(s, dtype=float); c = s[:144].reshape(12, 12).mean(0); return s - np.tile(c, len(s) // 12)
def _ss(emu, obs, base):              # skill score 1 - MSE_emu / MSE_baseline
    return float(1 - np.mean((emu - obs) ** 2) / np.mean((base - obs) ** 2))
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
