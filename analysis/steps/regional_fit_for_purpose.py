# Global and regional fit-for-purpose analysis (2022-2023): P1 global annual bias vs the spread of the training-period
# bias, P2 cross-model regional scatter statistics, P3 regional summary of monthly / anomaly skill; and the values of
# Supp Fig 2. Writes to OUT_DIR/fit_for_purpose/ and OUT_DIR/SuppFig2_values.csv.
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
AREA = seaflux.area.area_grid(lat, lon, return_dataarray=True).values           # as cal_mon
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})

# ---------------- RECCAP2 land-region mask ----------------
RMF = RECCAP_MASK
_r = xr.open_dataset(RMF)
print('RECCAP mask file attrs:', dict(_r.attrs)); print({k: dict(v.attrs) for k, v in _r.variables.items()})
_r = _r.reindex(latitude=_r['latitude'][::-1])
RID = _r['mask'].values.astype(float)
assert np.allclose(_r['latitude'].values, lat) and np.allclose(_r['longitude'].values, lon)
print('RECCAP IDs and cell counts:', {int(k): int((RID == k).sum()) for k in np.unique(RID[np.isfinite(RID)])})
REGIONS = REGION_SPEC
RMASK = {}
for nm, sp in REGIONS.items():
    m = np.isin(RID, sp['ids'])
    if 'lat_min' in sp: m &= ((lat >= sp['lat_min']) & (lat <= sp['lat_max']))[:, None]
    RMASK[nm] = m

# ---------------- global and regional monthly totals (GtC month-1) on each model's valid_mask_final cells ----------------
GM, GE, RM, RE, NC = {}, {}, {}, {}, {}
for n in NAMES:
    fm = final_masks[n].values
    for tag, src, G_, R_ in (('m', model_results, GM, RM), ('e', cnn_results, GE, RE)):
        F = src[n].values.astype(np.float64) * AREA[None] * SEC[:, None, None] / 1e12
        F[:, ~fm] = np.nan
        G_[n] = np.nansum(F, axis=(1, 2))
        R_[n] = {nm: np.nansum(F[:, m], axis=1) for nm, m in RMASK.items()}
    NC[n] = {nm: int((m & fm).sum()) for nm, m in RMASK.items()}
gchk = max(float(np.max(np.abs(GM[n] - cal_mon(model_results[n]).values))) for n in NAMES[:3])
print('global totals vs cal_mon, max abs diff (3 models):', gchk)

def r2(a, b): return pearson_r2(np.asarray(a), np.asarray(b))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(b) - np.asarray(a)) ** 2)))
def ann(s): return np.asarray(s).reshape(14, 12).sum(1)                       # 2010..2023
def own_anom(s): s = np.asarray(s); c = s[:144].reshape(12, 12).mean(0); return s - np.tile(c, 14)
def sc_test(s): return np.asarray(s)[144:].reshape(2, 12).mean(0)             # 2022-2023 mean seasonal cycle
def q_anom(s):                                                               # Fig 3 quarterly anomalies (QS-DEC), 8 values
    da = xr.DataArray(np.asarray(s), coords={'time': model_results[NAMES[0]].time.values}, dims='time')
    q = da.resample(time='QS-DEC').mean()
    return (q.sel(time=TEST_SLICE).groupby('time.season') - q.sel(time=REF_SLICE).groupby('time.season').mean('time')).values

# ================= Part 1: emulator error vs inter-model spread (global) =================
rows = []
for n in NAMES:
    ym, ye = ann(GM[n]), ann(GE[n])
    t = np.arange(12); res = ym[:12] - np.polyval(np.polyfit(t, ym[:12], 1), t)
    iav, iav_dt = float(np.std(ym[:12], ddof=1)), float(np.std(res, ddof=1))
    r = {'Model': disp(n)}
    for k, y in ((12, 2022), (13, 2023)):
        b = ye[k] - ym[k]
        r.update({f'Model_NBP_{y}': ym[k], f'Emulator_NBP_{y}': ye[k], f'Bias_{y}': b, f'AbsBias_{y}': abs(b),
                  f'AbsBias_{y}_over_IAV': abs(b) / iav, f'AbsBias_{y}_over_IAV_detrended': abs(b) / iav_dt})
    dm, de = ym[13] - ym[12], ye[13] - ye[12]
    r.update({'Change_2023_minus_2022_model': dm, 'Change_2023_minus_2022_emulator': de, 'Change_error': de - dm, 'AbsChange_error': abs(de - dm),
              'IAV_2010_2021_sd': iav, 'IAV_2010_2021_sd_detrended': iav_dt})
    rows.append(r)
P1 = pd.DataFrame(rows).set_index('Model')
st = {}
for c in P1.columns:
    v = P1[c].values.astype(float)
    st[c] = {'Inter-model SD (ddof=1)': np.std(v, ddof=1), 'Min': v.min(), 'Max': v.max(), 'Range': v.max() - v.min(),
             'Median': np.median(v), 'P25': np.percentile(v, 25), 'P75': np.percentile(v, 75)}
ST = pd.DataFrame(st)
P1x = pd.concat([P1, ST]); P1x.index.name = 'Model'
P1x.to_csv(OUT + 'P1_global_bias_vs_spread.csv', float_format='%.3f')
summ = []
for y in (2022, 2023):
    sd = float(np.std(P1[f'Model_NBP_{y}'], ddof=1)); ab = P1[f'AbsBias_{y}']
    summ.append({'item': f'{y} annual NBP', 'inter_model_sd': sd, 'model_min': P1[f'Model_NBP_{y}'].min(), 'model_max': P1[f'Model_NBP_{y}'].max(),
                 'model_range': P1[f'Model_NBP_{y}'].max() - P1[f'Model_NBP_{y}'].min(), 'absbias_median': ab.median(),
                 'absbias_p25': np.percentile(ab, 25), 'absbias_p75': np.percentile(ab, 75), 'ratio_median_absbias_over_sd': ab.median() / sd,
                 'n_absbias_lt_sd': int((ab < sd).sum()), 'median_absbias_over_IAV': P1[f'AbsBias_{y}_over_IAV'].median(),
                 'median_absbias_over_IAV_detrended': P1[f'AbsBias_{y}_over_IAV_detrended'].median()})
sd = float(np.std(P1['Change_2023_minus_2022_model'], ddof=1)); ae = P1['AbsChange_error']
summ.append({'item': '2023 minus 2022 change', 'inter_model_sd': sd, 'model_min': P1['Change_2023_minus_2022_model'].min(),
             'model_max': P1['Change_2023_minus_2022_model'].max(), 'model_range': np.ptp(P1['Change_2023_minus_2022_model']),
             'absbias_median': ae.median(), 'absbias_p25': np.percentile(ae, 25), 'absbias_p75': np.percentile(ae, 75),
             'ratio_median_absbias_over_sd': ae.median() / sd, 'n_absbias_lt_sd': int((ae < sd).sum())})
pd.DataFrame(summ).to_csv(OUT + 'P1_global_summary.csv', index=False, float_format='%.3f')
print(pd.DataFrame(summ).round(3).to_string())

# ================= Part 2: cross-model scatter (emulator vs model, one point per model) =================
def qvals(sm, se):
    return {'mean_annual_2022_2023': (ann(sm)[12:].mean(), ann(se)[12:].mean()),
            'seasonal_amplitude_2022_2023': (np.ptp(sc_test(sm)), np.ptp(sc_test(se))),
            'change_2023_minus_2022': (ann(sm)[13] - ann(sm)[12], ann(se)[13] - ann(se)[12])}
def xstats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); b, a = np.polyfit(x, y, 1); e = rmse(x, y); s = float(np.std(x, ddof=1))
    return {'n': len(x), 'R2': r2(x, y), 'slope': b, 'intercept': a, 'RMSE': e, 'inter_model_sd': s, 'RMSE_over_sd': e / s}
QN = ['mean_annual_2022_2023', 'seasonal_amplitude_2022_2023', 'change_2023_minus_2022']
UN = {'mean_annual_2022_2023': 'GtC yr-1', 'seasonal_amplitude_2022_2023': 'GtC month-1', 'change_2023_minus_2022': 'GtC yr-1'}
vals = {q: {'model': [], 'emulator': []} for q in QN}
for n in NAMES:
    v = qvals(GM[n], GE[n])
    for q in QN: vals[q]['model'].append(v[q][0]); vals[q]['emulator'].append(v[q][1])
V = pd.DataFrame({f'{q}_{k}': vals[q][k] for q in QN for k in ('model', 'emulator')}, index=D); V.index.name = 'Model'
V.to_csv(OUT + 'P2_global_scatter_values.csv', float_format='%.3f')
S2 = pd.DataFrame({q: xstats(vals[q]['model'], vals[q]['emulator']) for q in QN}).T; S2.index.name = 'quantity'
S2.to_csv(OUT + 'P2_global_scatter_stats.csv', float_format='%.3f')
print(S2.round(3).to_string())
TITLE = {'mean_annual_2022_2023': 'Mean annual NBP, 2022-2023', 'seasonal_amplitude_2022_2023': 'Seasonal amplitude, 2022-2023',
         'change_2023_minus_2022': 'Change 2023 minus 2022'}
def scat(ax, x, y, labels, title, unit, s):
    x, y = np.asarray(x), np.asarray(y); lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.6, zorder=0)
    ax.scatter(x, y, s=10, color='#1f4e79', zorder=2)
    for xi, yi, l in zip(x, y, labels): ax.annotate(l, (xi, yi), fontsize=4.5, xytext=(2, 1), textcoords='offset points')
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect('equal')
    ax.set_title(title, fontsize=7, fontweight='bold'); ax.set_xlabel(f'Model ({unit})'); ax.set_ylabel(f'Emulator ({unit})')
    ax.text(0.03, 0.97, f"$R^2$ = {s['R2']:.2f}\nslope = {s['slope']:.2f}\nRMSE = {s['RMSE']:.2f}", transform=ax.transAxes, va='top', fontsize=5.5)
    ax.tick_params(labelsize=5.5)
fig, axs = plt.subplots(1, 3, figsize=(183 / 25.4, 66 / 25.4), constrained_layout=True)
for ax, q in zip(axs, QN): scat(ax, vals[q]['model'], vals[q]['emulator'], D, TITLE[q], UN[q], S2.loc[q])
fig.savefig(OUT + 'P2_global_scatter_triptych.png', dpi=300); plt.close(fig)

# ================= Part 3: RECCAP2 regional statistics =================
rows = []
for n in NAMES:
    for nm in REGIONS:
        sm, se = RM[n][nm], RE[n][nm]
        am, ae_ = own_anom(sm), own_anom(se); cm, ce = sc_test(sm), sc_test(se); qm, qe = q_anom(sm), q_anom(se)
        ym, ye = ann(sm)[12:].mean(), ann(se)[12:].mean()
        rows.append({'Model': disp(n), 'Region': nm, 'Region_group': REGIONS[nm]['group'], 'N_valid_cells': NC[n][nm],
                     'Monthly_R2': r2(sm[144:], se[144:]), 'Monthly_RMSE': rmse(sm[144:], se[144:]),
                     'SeasonalCycle_R2': r2(cm, ce), 'SeasonalCycle_RMSE': rmse(cm, ce), 'SeasonalAmplitude_ratio_emu_over_model': np.ptp(ce) / np.ptp(cm),
                     'MonthlyAnomaly_R2': r2(am[144:], ae_[144:]), 'MonthlyAnomaly_RMSE': rmse(am[144:], ae_[144:]),
                     'QuarterlyAnomaly_R2': r2(qm, qe), 'QuarterlyAnomaly_n': int(len(qm)),
                     'MeanAnnual_2022_2023_model': ym, 'MeanAnnual_2022_2023_emulator': ye, 'MeanAnnual_2022_2023_diff': ye - ym,
                     'Annual_2022_model': ann(sm)[12], 'Annual_2022_emulator': ann(se)[12], 'Annual_2023_model': ann(sm)[13], 'Annual_2023_emulator': ann(se)[13]})
L3 = pd.DataFrame(rows); L3.to_csv(OUT + 'P3_regional_long.csv', index=False, float_format='%.4f')
MET = ['Monthly_R2', 'Monthly_RMSE', 'SeasonalCycle_R2', 'SeasonalCycle_RMSE', 'SeasonalAmplitude_ratio_emu_over_model', 'MonthlyAnomaly_R2',
       'MonthlyAnomaly_RMSE', 'QuarterlyAnomaly_R2', 'MeanAnnual_2022_2023_model', 'MeanAnnual_2022_2023_emulator', 'MeanAnnual_2022_2023_diff']
rows = []
for nm in REGIONS:
    g = L3[L3.Region == nm].set_index('Model')
    for mt in MET:
        rows.append({'Region': nm, 'metric': mt, 'median_19': g[mt].median(), 'min': g[mt].min(), 'min_model': g[mt].idxmin(),
                     'max': g[mt].max(), 'max_model': g[mt].idxmax()})
S3 = pd.DataFrame(rows); S3.to_csv(OUT + 'P3_regional_summary.csv', index=False, float_format='%.3f')
inv = pd.DataFrame([{'Region': nm, 'group': sp['group'], 'RECCAP_IDs': ' '.join(str(i) for i in sp['ids']),
                     'lat_limits': f"{sp.get('lat_min', '')} to {sp.get('lat_max', '')}" if 'lat_min' in sp else '',
                     'N_mask_cells': int(RMASK[nm].sum()), 'N_valid_cells_min_over_models': min(NC[n][nm] for n in NAMES),
                     'N_valid_cells_max_over_models': max(NC[n][nm] for n in NAMES)} for nm, sp in REGIONS.items()])
inv.to_csv(OUT + 'P3_region_inventory.csv', index=False)
print(inv.to_string())
H = S3.pivot(index='Region', columns='metric', values='median_19').loc[list(REGIONS), MET[:8]]
fig, ax = plt.subplots(figsize=(183 / 25.4, 0.32 * len(REGIONS) + 1.2), constrained_layout=True)
Z = (H - H.min()) / (H.max() - H.min())                                   # colour = per-column rank scale (draft)
ax.imshow(Z.values, cmap='Blues', aspect='auto', vmin=0, vmax=1.25)
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax.text(j, i, f'{H.values[i, j]:.2f}' if abs(H.values[i, j]) >= 0.01 else f'{H.values[i, j]:.3f}', ha='center', va='center', fontsize=5.5)
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=35, ha='right', fontsize=5.5)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=6)
ax.set_title('Median over 19 models, 2022-2023 (colour scaled per column; draft)', fontsize=7)
fig.savefig(OUT + 'P3_regional_median_heatmap.png', dpi=300); plt.close(fig)
# regional cross-model scatter for quantities a and b
rows = []
regs = [nm for nm in REGIONS]
fig, axs = plt.subplots(len(regs), 2, figsize=(120 / 25.4, 52 / 25.4 * len(regs)), constrained_layout=True)
for i, nm in enumerate(regs):
    for j, q in enumerate(QN[:2]):
        x = [qvals(RM[n][nm], RE[n][nm])[q][0] for n in NAMES]; y = [qvals(RM[n][nm], RE[n][nm])[q][1] for n in NAMES]
        s = xstats(x, y); rows.append({'Region': nm, 'quantity': q, **s})
        scat(axs[i, j], x, y, D, f'{nm}: {TITLE[q]}', UN[q], s)
fig.savefig(OUT + 'P2_regional_scatter_panels.png', dpi=200); plt.close(fig)
pd.DataFrame(rows).to_csv(OUT + 'P2_regional_scatter_stats.csv', index=False, float_format='%.3f')
print('written:', sorted(os.listdir(OUT)))

# ---------------- Supp Fig 2 values: cross-model scatter (one point per model), ten regions ----------------
R10 = [nm for nm, sp in REGIONS.items() if sp['group'] == 'RECCAP2']
assert len(R10) == 10
rows10 = []
for q in QN[:2]:
    for nm in R10:
        x = np.array([qvals(RM[n][nm], RE[n][nm])[q][0] for n in NAMES]); y = np.array([qvals(RM[n][nm], RE[n][nm])[q][1] for n in NAMES])
        rows10 += [{'Region': nm, 'quantity': q, 'Model': d, 'model': a, 'emulator': b} for d, a, b in zip(D, x, y)]
pd.DataFrame(rows10).to_csv(OUT_DIR + 'SuppFig2_values.csv', index=False, float_format='%.4f')
print('SuppFig2_values.csv written')
