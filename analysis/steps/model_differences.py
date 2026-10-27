# Why emulator skill differs between models: Fig 3 / Fig 4 values per model (Part 1) and their relation to features of
# the target model (Part 2: anomaly variability and persistence, local ERA5 regression skill, fire, PFT inputs, native
# grid; Mann-Whitney and Spearman tests with Benjamini-Hochberg correction). Writes to OUT_DIR/model_differences/.
# Runs after cells c24a, c24b and c21a.
ref_slice = REF_SLICE
test_slice = TEST_SLICE
# global annual fire emissions (fFire, PgC yr-1, 2010-2023) of each model, from the TRENDY v13 S2 fFire files
# (None = no fire module or no fFire output)
EMBED = {'fire_ann': {'CABLE-POP': None,
 'CARDAMOM': [1.247, 1.5601, 1.3305, 1.4202, 1.4909, 1.5433, 1.5074, 1.5055, 1.3726, 1.4911, 1.5343, 1.4439, 1.389,
              1.5352],
 'CLASSIC': [2.2647, 2.3688, 2.7421, 2.2389, 2.2084, 2.3599, 2.1281, 2.2868, 2.3617, 2.3426, 2.5363, 2.3508, 2.5608,
             2.4674],
 'CLM6.0': [2.7777, 2.6742, 3.1078, 2.7391, 2.7985, 3.1507, 2.7786, 2.8128, 2.8914, 3.0459, 2.9113, 2.8349, 2.8211,
            2.9769],
 'E3SM': None,
 'EDv3': [0.4455, 0.9998, 0.4493, 0.5878, 0.5052, 0.4466, 0.5965, 0.6422, 0.6155, 0.3317, 0.8756, 0.916, 0.9149,
          0.6305],
 'IBIS': None,
 'ISAM': None,
 'ISBA-CTRIP': [2.6658, 2.9064, 2.85, 2.6135, 2.7518, 2.7683, 2.5305, 2.8035, 2.7323, 2.8092, 2.7863, 2.7851, 2.7367,
                2.9311],
 'JSBACH': [1.6393, 1.6439, 1.6923, 1.7221, 1.5167, 1.6103, 1.6769, 1.582, 1.5616, 1.6389, 1.7317, 1.7304, 1.7254,
            1.8057],
 'JULES': [3.8034, 3.7232, 3.9737, 3.6038, 3.6942, 3.9792, 3.8017, 3.8763, 3.697, 3.8636, 3.9842, 3.7958, 3.8914,
           3.9828],
 'LPJwsl': [3.2471, 3.3212, 3.7294, 3.1962, 3.2439, 3.6673, 3.5147, 3.369, 3.2859, 3.4296, 3.4708, 3.7127, 3.477,
            3.6025],
 'LPX-Bern': [4.4055, 4.5482, 4.9227, 4.324, 4.463, 4.6933, 4.4894, 4.4361, 4.3251, 4.7499, 4.6375, 4.9666, 4.5912,
              4.7208],
 'OCN': None,
 'ORCHIDEE': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'SDGVM': [3.5934, 3.4894, 3.4464, 3.5374, 3.6912, 3.4429, 3.6186, 3.832, 3.6755, 3.3534, 3.5645, 3.72, 3.3027,
           3.3721],
 'VISIT': [1.6091, 1.8612, 2.6149, 2.0886, 2.1529, 2.5423, 2.3974, 2.0881, 2.15, 2.6035, 2.2649, 2.2646, 2.3855,
           2.5065],
 'VISIT-UT': [1467.4753, 1704.1819, 2289.2959, 1724.9111, 1869.7772, 2183.9915, 2034.9645, 1787.1052, 1733.8529,
              2182.6734, 1940.8708, 1956.5183, 1978.0262, 2065.1261],
 'iMAPLE': None}}
import os, re
from scipy import stats as _st
import pyarrow.feather as _pf
OUT = OUT_DIR + 'model_differences/'
os.makedirs(OUT, exist_ok=True)
NAMES = list(DGVM_list.values())                          # 19, DGVM_list order (101-119)
D = [disp(n) for n in NAMES]
F3 = lambda v: f'{v:.3f}'

# =============== Part 1.1  Fig 3 heatmap values (cell 24: df_r2, df_rmse, heatmap row order) ===============
fig2 = pd.concat([df_r2.add_prefix('R2_'), df_rmse.add_prefix('RMSE_')], axis=1); fig2.index.name = 'Model'
fig2.to_csv(OUT + 'Part1_Fig3_heatmap_values.csv', float_format='%.3f')
m19 = fig2.drop(index='Ensemble median')
summ = pd.DataFrame([{'column': c, 'median_19': m19[c].median(), 'min': m19[c].min(), 'min_model': m19[c].idxmin(),
                      'max': m19[c].max(), 'max_model': m19[c].idxmax(), 'Ensemble_median_row': fig2.loc['Ensemble median', c]}
                     for c in fig2.columns])
summ.to_csv(OUT + 'Part1_Fig3_column_summary.csv', index=False, float_format='%.3f')
print('Fig3 done', fig2.shape)

# =============== Part 1.2  Fig 4 and Supp Figs 3-5 panel statistics (cell 21/22: linregress on pooled valid samples) ===============
main_models = ['ORCHIDEE', 'CLM6.0']
si_models = [m for m in sorted(list(model_results.keys())) if m not in main_models]
where = {m: 'Fig 4' for m in main_models}
for k, b in enumerate([si_models[0:6], si_models[6:12], si_models[12:]]):
    for m in b: where[m] = f'Supp Fig {k + 3}'
COLS4 = ['Monthly NBP', 'Mean annual NBP', 'Mean seasonal cycle', 'Monthly anomalies']
rows = []
for n in NAMES:
    r = {'Model': disp(n), 'Figure': where[n]}
    for (x_raw, y_raw), c in zip(scatter_component_pairs(n), COLS4):
        msk = ~np.isnan(x_raw) & ~np.isnan(y_raw); x, y = x_raw[msk], y_raw[msk]
        sl, ic, rv, pv, se = linregress(x, y)
        r[f'R2_{c}'] = rv ** 2; r[f'RMSE_{c}'] = float(np.sqrt(np.mean((x - y) ** 2))); r[f'Slope_{c}'] = sl; r[f'n_{c}'] = int(x.size)
    rows.append(r)
fig4 = pd.DataFrame(rows).set_index('Model')
med = fig4.drop(columns='Figure').median(); med['Figure'] = ''
fig4.loc['Median of 19'] = med
fig4.to_csv(OUT + 'Part1_Fig4_SuppFig3-5_panel_stats.csv', float_format='%.3f')
print('Fig4 done')

# =============== Part 1.3  Fig 2 and Supp Fig 1 (cell 19/20: cal_mon global totals) ===============
rows = []
for n in NAMES:
    gm, gc = cal_mon(model_results[n]), cal_mon(cnn_results[n])
    t = pd.to_datetime(gm.time.values); te = t >= '2022-01-01'
    ym, ye = gm.values[te], gc.values[te]
    rows.append({'Model': disp(n), 'Figure': 'Fig 2' if n in main_models else 'Supp Fig 1',
                 'R2_global_monthly_NBP_2022_2023': pearson_r2(ym, ye), 'RMSE_GtC_month-1': float(np.sqrt(np.mean((ym - ye) ** 2))),
                 'Model_NBP_2022_GtC_yr-1': float(ym[:12].sum()), 'Emulator_NBP_2022_GtC_yr-1': float(ye[:12].sum()),
                 'Model_NBP_2023_GtC_yr-1': float(ym[12:].sum()), 'Emulator_NBP_2023_GtC_yr-1': float(ye[12:].sum())})
fig3 = pd.DataFrame(rows).set_index('Model')
med = fig3.drop(columns='Figure').median(); med['Figure'] = ''
fig3.loc['Median of 19'] = med
fig3.to_csv(OUT + 'Part1_Fig2_SuppFig1_global.csv', float_format='%.3f')
print('Fig2 done')

# =============== Part 2  skill metrics (n = 19) ===============
S = pd.DataFrame(index=D)
for c in df_r2.columns: S[f'R2_{c}'] = [fig2.loc[d, f'R2_{c}'] for d in D]
for c in df_rmse.columns: S[f'RMSE_{c}'] = [fig2.loc[d, f'RMSE_{c}'] for d in D]
t3 = pd.read_csv(OUT_DIR + 'SuppTable5.csv', index_col=0)
S['R2_global_monthly_anomaly_SuppTable5'] = [t3.loc[d, 'R2_global_monthly_anomaly_2022_2023'] for d in D]
tail = {}
for line in open(OUT_DIR + 'Fig6_tail_underestimation.txt'):
    p = line.split()
    if len(p) == 8 and p[1].isdigit(): tail[p[0]] = [float(v) for v in p[1:]]
S['Tail_underestimation_q01_pct'] = [tail[d][3] for d in D]; S['Tail_underestimation_q99_pct'] = [tail[d][6] for d in D]
p22, p23 = [], []
for n in NAMES:
    dm, dc = to_gC_month(model_results[n]), to_gC_month(cnn_results[n])
    am, ac = calculate_anomaly(dm.sel(time=TEST_SLICE), clim_ref(dm)), calculate_anomaly(dc.sel(time=TEST_SLICE), clim_ref(dc))
    p22.append(pearson_r2(am.sel(time='2022').values, ac.sel(time='2022').values)); p23.append(pearson_r2(am.sel(time='2023').values, ac.sel(time='2023').values))
S['Pooled_grid_anomaly_R2_2022'] = p22; S['Pooled_grid_anomaly_R2_2023'] = p23
rec = {}
for n in NAMES:
    _dm, _dc = to_gC_month(model_results[n]), to_gC_month(cnn_results[n])
    _am, _ac = calculate_anomaly(_dm, clim_ref(_dm)), calculate_anomaly(_dc, clim_ref(_dc))
    rec[n] = {y: pearson_r2(_am.sel(time=y).values, _ac.sel(time=y).values) for y in ('2021', '2022', '2023')}
del _dm, _dc, _am, _ac

# =============== Part 2  target-model features (model output, forcing and existing facts only) ===============
FE = pd.DataFrame(index=D)
# ERA5 t2m / tp from the paper's training input file, rows indexed by (latitude, longitude) as in training
TRAIN_FILE = DATA_FEATHER
def era5_cube(path, col):
    t = _pf.read_table(path, columns=['year', 'month', 'latitude', 'longitude', col], memory_map=True)
    y = t['year'].to_numpy(); m = t['month'].to_numpy()
    s = ((y - 2009) * 12 + m - 1 >= 9) & (y <= 2021)               # 2009-10 .. 2021-12 (147 months)
    ti = ((y[s] - 2009) * 12 + m[s] - 1 - 9).astype(int)
    a = np.full((147, 360, 720), np.nan, dtype=np.float64)
    a[ti, t['latitude'].to_numpy()[s].astype(int), t['longitude'].to_numpy()[s].astype(int)] = t[col].to_numpy(zero_copy_only=False)[s]
    return a
T2 = era5_cube(TRAIN_FILE, 't2m'); TP = era5_cube(TRAIN_FILE, 'tp')
def era_anoms(a):                                              # per cell, vs 2010-2021 monthly climatology; lag0 and mean of t-3..t-1
    clim = a[3:].reshape(12, 12, 360, 720).mean(0)
    an = a - np.concatenate([clim[9:], np.tile(clim, (12, 1, 1))])  # 147 months start in October
    lag0 = an[3:]; prev3 = (an[0:144] + an[1:145] + an[2:146]) / 3.0
    return lag0, prev3
T0a, T3a = era_anoms(T2); P0a, P3a = era_anoms(TP)
del T2, TP

PFT_CH = {'ORCHIDEE': 15, 'CABLE-POP': 10, 'CLASSIC': 10, 'IBIS': 10, 'JSBACH': 14, 'JULES': 17, 'LPX-Bern': 30, 'OCN': 13, 'VISIT': 16,
          'iMAPLE': 8, 'ISBA-CTRIP': 19, 'SDGVM': 16}             # PFT input channels of each checkpoint (input width - 10)
FIRE = {'ORCHIDEE': 0, 'CABLE-POP': 0, 'OCN': 0, 'iMAPLE': 0, 'ISAM': 0}   # models without fire; all others have fire (CARDAMOM included)
GRID = {'ORCHIDEE': (0.5, 0.5), 'CABLE-POP': (1.0, 1.0), 'CLASSIC': (1.0, 1.0), 'IBIS': (0.5, 0.5), 'JSBACH': (1.8652, 1.875),
        'JULES': (0.5, 0.5), 'LPX-Bern': (0.5, 0.5), 'OCN': (1.0, 1.0), 'VISIT': (0.5, 0.5), 'iMAPLE': (1.0, 1.0),
        'ISBA-CTRIP': (1.0, 1.0), 'CLM6.0': (0.9424, 1.25), 'LPJwsl': (0.5, 0.5), 'EDv3': (0.5, 0.5), 'ISAM': (0.5, 0.5),
        'E3SM': (0.9424, 1.25), 'CARDAMOM': (1.0, 1.0), 'SDGVM': (1.0, 1.0), 'visit-ut': (0.5, 0.5)}   # raw v13 S2 nbp files
cols = {k: [] for k in ['Anom_sd_pooled_2010_2021', 'Anom_sd_cellmedian_2010_2021', 'Anom_sd_pooled_2022_2023', 'Anom_sd_cellmedian_2022_2023',
                        'Anom_var_fraction_cellmedian_2010_2021', 'Anom_lag1_autocorr_cellmedian_2010_2021',
                        'ERA5_local_OLS_R2_cellmedian_2010_2021', 'ERA5_OLS_n_cells', 'N_valid_cells']}
for n in NAMES:
    msk = final_masks[n].values
    X = to_gC_month(model_results[n]).values[:, msk].astype(np.float64)        # 168 x cells, gC m-2 month-1
    clim = X[:144].reshape(12, 12, -1).mean(0); A = X - np.tile(clim, (14, 1))
    a1, a2 = A[:144], A[144:]
    cols['Anom_sd_pooled_2010_2021'].append(float(np.std(a1))); cols['Anom_sd_cellmedian_2010_2021'].append(float(np.median(a1.std(0))))
    cols['Anom_sd_pooled_2022_2023'].append(float(np.std(a2))); cols['Anom_sd_cellmedian_2022_2023'].append(float(np.median(a2.std(0))))
    cols['Anom_var_fraction_cellmedian_2010_2021'].append(float(np.median(a1.var(0) / X[:144].var(0))))
    u, v = a1[1:] - a1[1:].mean(0), a1[:-1] - a1[:-1].mean(0)
    cols['Anom_lag1_autocorr_cellmedian_2010_2021'].append(float(np.median((u * v).sum(0) / np.sqrt((u * u).sum(0) * (v * v).sum(0)))))
    # per-cell OLS: model anomaly ~ 1 + t2m(t) + t2m(mean t-3..t-1) + tp(t) + tp(mean t-3..t-1), 2010-2021
    Z = np.stack([np.ones((144, int(msk.sum()))), T0a[:, msk], T3a[:, msk], P0a[:, msk], P3a[:, msk]], axis=2)
    ok = np.isfinite(Z).all(axis=(0, 2))
    Zk, yk = Z[:, ok], a1[:, ok]
    XtX = np.einsum('tci,tcj->cij', Zk, Zk); Xty = np.einsum('tci,tc->ci', Zk, yk)
    beta = np.einsum('cij,cj->ci', np.linalg.pinv(XtX), Xty)
    res = yk - np.einsum('tci,ci->tc', Zk, beta)
    r2 = 1 - (res ** 2).sum(0) / ((yk - yk.mean(0)) ** 2).sum(0)
    cols['ERA5_local_OLS_R2_cellmedian_2010_2021'].append(float(np.median(r2))); cols['ERA5_OLS_n_cells'].append(int(ok.sum()))
    cols['N_valid_cells'].append(int(msk.sum()))
for k, v in cols.items(): FE[k] = v
FE['Ref_q01_2022_2023'] = [tail[d][1] for d in D]; FE['Ref_q99_2022_2023'] = [tail[d][4] for d in D]
FE['Ref_tail_width_q99_minus_q01'] = FE['Ref_q99_2022_2023'] - FE['Ref_q01_2022_2023']
FE['PFT_input'] = [int(n in PFT_CH) for n in NAMES]; FE['PFT_channels'] = [PFT_CH.get(n, 0) for n in NAMES]
FE['Fire'] = [FIRE.get(n, 1) for n in NAMES]
fa = EMBED['fire_ann']; key = {n: ('VISIT-UT' if n == 'visit-ut' else n) for n in NAMES}
def _fire(n):
    s = fa.get(key[n])
    if not s or FE.loc[disp(n), 'Fire'] == 0: return np.nan, np.nan, np.nan
    s = np.asarray(s, float) / (1000.0 if n == 'visit-ut' else 1.0)    # VISIT-UT fFire labelled kg but ~1000x too large: treated as g
    return float(s.mean()), float(s.std(ddof=1)), float(s.std(ddof=1) / s.mean())
fr = [_fire(n) for n in NAMES]
FE['Fire_mean_GtC_yr'] = [x[0] for x in fr]; FE['Fire_IAV_sd_GtC_yr'] = [x[1] for x in fr]; FE['Fire_IAV_CV'] = [x[2] for x in fr]
FE['Native_dlat_deg'] = [GRID[n][0] for n in NAMES]; FE['Native_dlon_deg'] = [GRID[n][1] for n in NAMES]
FE['Native_grid_spacing_deg'] = [float(np.sqrt(GRID[n][0] * GRID[n][1])) for n in NAMES]
one = {n: rec[n]['2021'] for n in NAMES}   # the 2010-2021 rows of the recursive files are the one-step output
FE['OneStep_2021_pooled_anom_R2'] = [one.get(n) if one.get(n) is not None else np.nan for n in NAMES]
FE['InSample_2021_pooled_anom_R2_paper_files'] = [rec[n]['2021'] for n in NAMES]
print('features done')

TAB = pd.concat([S, FE], axis=1); TAB.index.name = 'Model'
TAB.to_csv(OUT + 'Part2_table_19_models.csv', float_format='%.4f')

# =============== Part 2  statistics ===============
SK = list(S.columns)
CONT = ['Anom_sd_pooled_2010_2021', 'Anom_sd_cellmedian_2010_2021', 'Anom_sd_pooled_2022_2023', 'Anom_sd_cellmedian_2022_2023',
        'Anom_var_fraction_cellmedian_2010_2021', 'Anom_lag1_autocorr_cellmedian_2010_2021', 'ERA5_local_OLS_R2_cellmedian_2010_2021',
        'Ref_q01_2022_2023', 'Ref_q99_2022_2023', 'Ref_tail_width_q99_minus_q01', 'PFT_channels_PFTmodels_only',
        'Fire_IAV_sd_GtC_yr', 'Fire_IAV_CV', 'N_valid_cells', 'Native_grid_spacing_deg', 'OneStep_2021_pooled_anom_R2',
        'InSample_2021_pooled_anom_R2_paper_files']
TAB['PFT_channels_PFTmodels_only'] = TAB['PFT_channels'].where(TAB['PFT_input'] == 1)
def _sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5: return np.nan, np.nan, int(ok.sum())
    r, p = _st.spearmanr(a[ok], b[ok]); return float(r), float(p), int(ok.sum())
rows = []
for s_ in SK:
    for f_ in CONT:
        r, p, k = _sp(TAB[s_].values.astype(float), TAB[f_].values.astype(float))
        rows.append({'skill_metric': s_, 'feature': f_, 'spearman_rho': r, 'p_value': p, 'n': k})
SP = pd.DataFrame(rows)
pv = SP['p_value'].values; okp = np.isfinite(pv); q = np.full(pv.shape, np.nan)   # Benjamini-Hochberg over all tests in this table
o = np.argsort(pv[okp]); m_ = okp.sum(); qq = pv[okp][o] * m_ / np.arange(1, m_ + 1); qq = np.minimum.accumulate(qq[::-1])[::-1]
tmp = np.empty(m_); tmp[o] = np.minimum(qq, 1); q[okp] = tmp; SP['BH_q_value'] = q
SP.to_csv(OUT + 'Part2_spearman_skill_vs_features_long.csv', index=False, float_format='%.4f')
SP.pivot(index='skill_metric', columns='feature', values='spearman_rho').loc[SK, CONT].to_csv(OUT + 'Part2_spearman_rho_matrix.csv', float_format='%.3f')
SP.pivot(index='skill_metric', columns='feature', values='p_value').loc[SK, CONT].to_csv(OUT + 'Part2_spearman_p_matrix.csv', float_format='%.4f')
rows = []
for b_ in ['PFT_input', 'Fire']:
    g1, g0 = TAB[TAB[b_] == 1], TAB[TAB[b_] == 0]
    for s_ in SK:
        u, p = _st.mannwhitneyu(g1[s_], g0[s_], alternative='two-sided')
        rows.append({'binary_feature': b_, 'skill_metric': s_, 'n_yes': len(g1), 'n_no': len(g0), 'median_yes': g1[s_].median(),
                     'median_no': g0[s_].median(), 'U': float(u), 'p_value': float(p)})
MW = pd.DataFrame(rows); MW.to_csv(OUT + 'Part2_mannwhitney_binary.csv', index=False, float_format='%.4f')
# relations among the features themselves
FF = []
for i, a in enumerate(CONT):
    for b in CONT[i + 1:]:
        r, p, k = _sp(TAB[a].values.astype(float), TAB[b].values.astype(float)); FF.append({'feature_a': a, 'feature_b': b, 'spearman_rho': r, 'p_value': p, 'n': k})
pd.DataFrame(FF).to_csv(OUT + 'Part2_feature_feature_spearman.csv', index=False, float_format='%.4f')
rows = []
for b_ in ['PFT_input', 'Fire']:
    for f_ in [c for c in CONT if c != 'PFT_channels_PFTmodels_only']:
        g1, g0 = TAB.loc[TAB[b_] == 1, f_].dropna(), TAB.loc[TAB[b_] == 0, f_].dropna()
        if len(g1) >= 2 and len(g0) >= 2:
            u, p = _st.mannwhitneyu(g1, g0, alternative='two-sided')
            rows.append({'binary_feature': b_, 'feature': f_, 'n_yes': len(g1), 'n_no': len(g0), 'median_yes': g1.median(), 'median_no': g0.median(), 'U': float(u), 'p_value': float(p)})
pd.DataFrame(rows).to_csv(OUT + 'Part2_feature_by_binary_group.csv', index=False, float_format='%.4f')
ct = pd.crosstab(TAB['PFT_input'], TAB['Fire']); odds, pf_ = _st.fisher_exact(ct.values)
with open(OUT + 'Part2_PFT_x_Fire_crosstab.txt', 'w') as fh:
    fh.write('rows = PFT_input (0/1), columns = Fire (0/1), n = 19\n' + ct.to_string() + f'\nFisher exact two-sided p = {pf_:.4f}\n')
    fh.write('PFT=1, Fire=0: ' + ', '.join(TAB.index[(TAB.PFT_input == 1) & (TAB.Fire == 0)]) + '\n')
    fh.write('PFT=0, Fire=0: ' + ', '.join(TAB.index[(TAB.PFT_input == 0) & (TAB.Fire == 0)]) + '\n')

# =============== scatter plots for the strongest pairs ===============
top = SP.dropna(subset=['spearman_rho']).reindex(SP['spearman_rho'].abs().sort_values(ascending=False).index).head(5)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})
fig, axs = plt.subplots(1, 5, figsize=(183 / 25.4, 50 / 25.4), constrained_layout=True)
for ax, (_, r) in zip(axs, top.iterrows()):
    x, y = TAB[r.feature].astype(float), TAB[r.skill_metric].astype(float); ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=8, color='#1f4e79')
    for nm in TAB.index[ok]:
        ax.annotate(nm, (x[nm], y[nm]), fontsize=4.5, xytext=(2, 1), textcoords='offset points')
    ax.set_xlabel(r.feature, fontsize=5.5); ax.set_ylabel(r.skill_metric, fontsize=5.5)
    ax.set_title(f"rho = {r.spearman_rho:.2f}, p = {r.p_value:.1e}, n = {int(r.n)}", fontsize=5.5)
    ax.tick_params(labelsize=5)
fig.savefig(OUT + 'Part2_scatter_top5.png', dpi=300)
for k, (_, r) in enumerate(top.iterrows(), 1):
    f2, ax = plt.subplots(figsize=(3.2, 3.0), constrained_layout=True)
    x, y = TAB[r.feature].astype(float), TAB[r.skill_metric].astype(float); ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=14, color='#1f4e79')
    for nm in TAB.index[ok]:
        ax.annotate(nm, (x[nm], y[nm]), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel(r.feature); ax.set_ylabel(r.skill_metric)
    ax.set_title(f"Spearman rho = {r.spearman_rho:.2f}, p = {r.p_value:.1e}, n = {int(r.n)}", fontsize=7)
    f2.savefig(OUT + f'Part2_scatter_{k}.png', dpi=300); plt.close(f2)
print(top.to_string())
print('written:', sorted(os.listdir(OUT)))
