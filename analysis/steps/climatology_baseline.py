# Emulator vs climatology baseline (each model's own 2010-2021 climatology) at global, regional and grid scale, 2022-2023:
# P4_1 grid mean-annual map, P4_2 monthly totals, P4_3 annual errors, P4_4 anomaly skill vs zero anomaly.
# Writes P4_* files to OUT_DIR/fit_for_purpose/.
# Regions: the ten RECCAP2 land regions as grouped in the RECCAP2 land map of the NRT carbon budget project, from the
# 25 sub-regions of Land_reccap_regions_mask_05x05.nc. North America includes Greenland, Iceland, the Faroe Islands and
# Svalbard (id 3); Latin America is South America plus Mexico, Central America and the Caribbean (ids 17-21);
# 'subset' entries are tropical subsets (23.5S-23.5N).
REGION_SPEC = {
    'North America': {'ids': [1, 2, 3], 'group': 'RECCAP2'},
    'Europe': {'ids': [4, 5, 6, 8, 9], 'group': 'RECCAP2'},
    'West Asia': {'ids': [7, 11], 'group': 'RECCAP2'},
    'Russia': {'ids': [10], 'group': 'RECCAP2'},
    'East Asia': {'ids': [12, 13, 14], 'group': 'RECCAP2'},
    'South Asia': {'ids': [15], 'group': 'RECCAP2'},
    'Southeast Asia': {'ids': [16], 'group': 'RECCAP2'},
    'Latin America': {'ids': [17, 18, 19, 20, 21], 'group': 'RECCAP2'},
    'Oceania': {'ids': [22], 'group': 'RECCAP2'},
    'Africa': {'ids': [23, 24, 25], 'group': 'RECCAP2'},
    'Tropical Americas': {'ids': [17, 18, 19, 20, 21], 'lat_min': -23.5, 'lat_max': 23.5, 'group': 'subset'},
    'Tropical Africa': {'ids': [23, 24, 25], 'lat_min': -23.5, 'lat_max': 23.5, 'group': 'subset'},
    'Tropics (all RECCAP land)': {'ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 'lat_min': -23.5, 'lat_max': 23.5, 'group': 'subset'},
}
import os
OUT = OUT_DIR + 'fit_for_purpose/'
os.makedirs(OUT, exist_ok=True)
NAMES = list(DGVM_list.values()); D = [disp(n) for n in NAMES]
TIME = pd.DatetimeIndex(pd.to_datetime(model_results[NAMES[0]].time.values))
assert len(TIME) == 168 and TIME[0].year == 2010 and TIME[-1].year == 2023
SEC = TIME.days_in_month.values * 86400.0
lat = model_results[NAMES[0]].lat.values; lon = model_results[NAMES[0]].lon.values
AREA = seaflux.area.area_grid(lat, lon, return_dataarray=True).values
_r = xr.open_dataset(RECCAP_MASK); _r = _r.reindex(latitude=_r['latitude'][::-1])
RID = _r['mask'].values.astype(float)
assert np.allclose(_r['latitude'].values, lat) and np.allclose(_r['longitude'].values, lon)
REGIONS = REGION_SPEC
RMASK = {}
for nm, sp in REGIONS.items():
    m = np.isin(RID, sp['ids'])
    if 'lat_min' in sp: m &= ((lat >= sp['lat_min']) & (lat <= sp['lat_max']))[:, None]
    RMASK[nm] = m
def r2(a, b): return pearson_r2(np.asarray(a), np.asarray(b))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(b) - np.asarray(a)) ** 2)))
def ann(s): return np.asarray(s).reshape(14, 12).sum(1)
def clim12(s): return np.asarray(s)[:144].reshape(12, 12).mean(0)
def own_anom(s): s = np.asarray(s); return s - np.tile(clim12(s), 14)
def ss0(emu_a, mod_a): emu_a, mod_a = np.asarray(emu_a), np.asarray(mod_a); return float(1 - np.sum((emu_a - mod_a) ** 2) / np.sum(mod_a ** 2))

rows_grid, rows_mon, rows_ann, rows_ss = [], [], [], []
for n in NAMES:
    fm = final_masks[n].values
    Mk = model_results[n].values.astype(np.float64); Ek = cnn_results[n].values.astype(np.float64)
    # ---- 1. grid annual maps (gC m-2 yr-1), same construction as cell 24: annual totals with 12 months, mean over years ----
    gm = Mk * SEC[:, None, None] * 1000.0; ge = Ek * SEC[:, None, None] * 1000.0
    ya_m, ya_e = gm.reshape(14, 12, 360, 720).sum(1), ge.reshape(14, 12, 360, 720).sum(1)      # NaN outside the valid mask
    clim_map, test_map, emu_map = ya_m[:12].mean(0), ya_m[12:].mean(0), ya_e[12:].mean(0)
    ok = fm & np.isfinite(clim_map) & np.isfinite(test_map) & np.isfinite(emu_map)
    rows_grid.append({'Model': disp(n), 'n_cells': int(ok.sum()),
                      'R2_climatology_map_vs_2022_2023': r2(clim_map[ok], test_map[ok]), 'RMSE_climatology_gC_m2_yr': rmse(test_map[ok], clim_map[ok]),
                      'R2_emulator_map_vs_2022_2023': r2(emu_map[ok], test_map[ok]), 'RMSE_emulator_gC_m2_yr': rmse(test_map[ok], emu_map[ok])})
    # ---- 4b. grid pooled anomaly skill vs zero (all valid cells x 24 months, unweighted) ----
    cm, ce = gm[:144].reshape(12, 12, 360, 720).mean(0), ge[:144].reshape(12, 12, 360, 720).mean(0)
    am, ae = gm[144:] - np.tile(cm, (2, 1, 1)), ge[144:] - np.tile(ce, (2, 1, 1))
    x, y = am[:, fm].ravel(), ae[:, fm].ravel(); k = np.isfinite(x) & np.isfinite(y)
    rows_ss.append({'Model': disp(n), 'Region': 'Grid cells (pooled)', 'n_values': int(k.sum()), 'SS_vs_zero_anomaly': ss0(y[k], x[k])})
    # ---- totals: global and 13 regions (GtC month-1) on the model's own valid cells ----
    F_m = Mk * AREA[None] * SEC[:, None, None] / 1e12; F_e = Ek * AREA[None] * SEC[:, None, None] / 1e12
    F_m[:, ~fm] = np.nan; F_e[:, ~fm] = np.nan
    series = {'Global': (np.nansum(F_m, axis=(1, 2)), np.nansum(F_e, axis=(1, 2)))}
    for nm, m in RMASK.items(): series[nm] = (np.nansum(F_m[:, m], axis=1), np.nansum(F_e[:, m], axis=1))
    del F_m, F_e, gm, ge, ya_m, ya_e
    for nm, (sm, se) in series.items():
        clim_pred = np.tile(clim12(sm), 2)                                   # 2010-2021 monthly climatology as the 2022-2023 forecast
        rows_mon.append({'Model': disp(n), 'Region': nm, 'R2_climatology': r2(clim_pred, sm[144:]), 'RMSE_climatology': rmse(sm[144:], clim_pred),
                         'R2_emulator': r2(se[144:], sm[144:]), 'RMSE_emulator': rmse(sm[144:], se[144:])})
        ym, ye = ann(sm), ann(se); base = ym[:12].mean()
        for k_, yr in ((12, 2022), (13, 2023)):
            rows_ann.append({'Model': disp(n), 'Region': nm, 'Year': yr, 'Model_annual': ym[k_], 'Climatology_forecast': base,
                             'AbsErr_climatology': abs(base - ym[k_]), 'Emulator_annual': ye[k_], 'AbsErr_emulator': abs(ye[k_] - ym[k_])})
        am_, ae_ = own_anom(sm)[144:], own_anom(se)[144:]
        rows_ss.append({'Model': disp(n), 'Region': nm, 'n_values': 24, 'SS_vs_zero_anomaly': ss0(ae_, am_)})
    print(n, flush=True)

gchk = max(abs(r['R2_emulator_map_vs_2022_2023'] - pearson_r2(annual_gC(cnn_results[NAMES[i]].sel(time=TEST_SLICE)).mean('year').values.ravel(),
                                                             annual_gC(model_results[NAMES[i]].sel(time=TEST_SLICE)).mean('year').values.ravel()))
           for i, r in enumerate(rows_grid[:3]))
print('emulator mean-annual map R2 vs cell-24 construction (3 models), max abs diff:', gchk)
def with_median(df, cols):
    med = df[cols].median(); med['Model'] = 'Median of 19'; return pd.concat([df, pd.DataFrame([med])], ignore_index=True)
G1 = with_median(pd.DataFrame(rows_grid), ['n_cells', 'R2_climatology_map_vs_2022_2023', 'RMSE_climatology_gC_m2_yr', 'R2_emulator_map_vs_2022_2023', 'RMSE_emulator_gC_m2_yr'])
G1.to_csv(OUT + 'P4_1_grid_mean_annual_map.csv', index=False, float_format='%.3f')
M2 = pd.DataFrame(rows_mon); M2.to_csv(OUT + 'P4_2_monthly_total_long.csv', index=False, float_format='%.4f')
S2 = M2.groupby('Region', sort=False)[['R2_climatology', 'RMSE_climatology', 'R2_emulator', 'RMSE_emulator']].median().reset_index()
S2['n_models_emulator_R2_gt_climatology_R2'] = M2.assign(b=M2.R2_emulator > M2.R2_climatology).groupby('Region', sort=False)['b'].sum().values
S2.to_csv(OUT + 'P4_2_monthly_total_median.csv', index=False, float_format='%.3f')
A3 = pd.DataFrame(rows_ann); A3.to_csv(OUT + 'P4_3_annual_error_long.csv', index=False, float_format='%.4f')
S3 = []
for (nm, yr), g in A3.groupby(['Region', 'Year'], sort=False):
    S3.append({'Region': nm, 'Year': yr, 'median_AbsErr_climatology': g.AbsErr_climatology.median(), 'median_AbsErr_emulator': g.AbsErr_emulator.median(),
               'n_AbsErr_climatology_lt_0.4': int((g.AbsErr_climatology < 0.4).sum()), 'n_AbsErr_emulator_lt_0.4': int((g.AbsErr_emulator < 0.4).sum()),
               'n_emulator_better': int((g.AbsErr_emulator < g.AbsErr_climatology).sum())})
pd.DataFrame(S3).to_csv(OUT + 'P4_3_annual_error_summary.csv', index=False, float_format='%.3f')
S4 = pd.DataFrame(rows_ss); S4.to_csv(OUT + 'P4_4_anomaly_skill_vs_zero_long.csv', index=False, float_format='%.4f')
S4m = S4.groupby('Region', sort=False)['SS_vs_zero_anomaly'].agg(['median', 'min', 'max']).reset_index()
S4m['n_models_SS_gt_0'] = S4.assign(b=S4.SS_vs_zero_anomaly > 0).groupby('Region', sort=False)['b'].sum().values
S4m.to_csv(OUT + 'P4_4_anomaly_skill_vs_zero_median.csv', index=False, float_format='%.3f')
pd.set_option('display.width', 250)
print(G1.round(3).to_string()); print(S2.round(3).to_string()); print(pd.DataFrame(S3).round(3).to_string()); print(S4m.round(3).to_string())
print('written:', sorted(f for f in os.listdir(OUT) if f.startswith('P4_')))
