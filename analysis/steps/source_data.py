# Source data of the figures and tables, as CSV files in OUT_DIR/source_data/. Statistics are copied from the outputs of
# the other jobs; global monthly totals, the Q-Q quantiles of Fig 6 / Supp Figs 7-9 and the Supp Fig 6 statistics are
# computed here with the definitions of the figure code (cells/c19, c20, c23, c25). Gridded TRENDY values are not
# written: only global totals and statistics.
import os
import re
import shutil
from scipy.stats import linregress

SD = OUT_DIR + 'source_data/'
os.makedirs(SD, exist_ok=True)
NAMES = list(DGVM_list.values())

# ---- Fig 2 and Supp Fig 1: global monthly NBP totals 2010-2023 (GtC month-1), model and emulator
rows = []
for n in NAMES:
    gm, ge = cal_mon(model_results[n]), cal_mon(cnn_results[n])
    t = pd.DatetimeIndex(pd.to_datetime(gm.time.values))
    for k in range(len(t)):
        rows.append({'Model': disp(n), 'Year': t[k].year, 'Month': t[k].month,
                     'Model_NBP_GtC_month-1': float(gm.values[k]), 'Emulator_NBP_GtC_month-1': float(ge.values[k])})
pd.DataFrame(rows).to_csv(SD + 'Fig2_SuppFig1_global_monthly_NBP.csv', index=False, float_format='%.5f')
shutil.copy(OUT_DIR + 'model_differences/Part1_Fig2_SuppFig1_global.csv', SD + 'Fig2_SuppFig1_statistics.csv')

# ---- Fig 3: R2 and RMSE of each model (heat map values)
f3 = pd.read_csv(OUT_DIR + 'model_differences/Part1_Fig3_heatmap_values.csv')
f3 = f3.rename(columns={c: c + (' (gC m-2 yr-1)' if 'Mean annual' in c else ' (gC m-2 month-1)') for c in f3.columns if c.startswith('RMSE_')})
f3.to_csv(SD + 'Fig3_R2_RMSE_by_model.csv', index=False, float_format='%.3f')

# ---- Fig 4 and Supp Figs 3-5: statistics of each panel (grid-cell scatter, 2022-2023)
p4 = pd.read_csv(OUT_DIR + 'model_differences/Part1_Fig4_SuppFig3-5_panel_stats.csv')
for c in [c for c in p4.columns if c.startswith('n_')]:
    p4[c] = p4[c].astype(int)
p4.to_csv(SD + 'Fig4_SuppFig3-5_panel_statistics.csv', index=False, float_format='%.3f')

# ---- Fig 5: spatial R2 of the maps
rows = []
for line in open(OUT_DIR + 'Fig5_spatial_R2.txt'):
    m = re.match(r'\s*(\S+)\s+(Mean annual|JFM anomaly|JAS anomaly)\s+([-\d.]+)\s+(\d+)\s*$', line)
    if m:
        rows.append({'Model': m.group(1), 'Map': m.group(2), 'Spatial_R2': float(m.group(3)), 'n_cells': int(m.group(4))})
assert len(rows) == 6
pd.DataFrame(rows).to_csv(SD + 'Fig5_spatial_R2.csv', index=False, float_format='%.3f')

# ---- Supp Fig 6: grid-cell JFM and JAS mean anomalies 2022-2023, ORCHIDEE and CLM6.0 (as cells/c23_suppfig6.py)
rows = []
for n in ['ORCHIDEE', 'CLM6.0']:
    dm, dc = to_gC_month(model_results[n]), to_gC_month(cnn_results[n])
    am = calculate_anomaly(dm.sel(time=TEST_SLICE), clim_ref(dm))
    ac = calculate_anomaly(dc.sel(time=TEST_SLICE), clim_ref(dc))
    for season, months in (('JFM', [1, 2, 3]), ('JAS', [7, 8, 9])):
        sel = am.time.dt.month.isin(months)
        x = am.where(sel).mean('time').values.flatten(); y = ac.where(sel).mean('time').values.flatten()
        ok = ~np.isnan(x) & ~np.isnan(y); x, y = x[ok], y[ok]
        r = linregress(x, y)
        rows.append({'Model': disp(n), 'Season': season, 'n_cells': int(ok.sum()), 'R2': r.rvalue ** 2,
                     'RMSE_gC_m-2_month-1': float(np.sqrt(np.mean((x - y) ** 2))), 'Slope': r.slope, 'Intercept': r.intercept})
pd.DataFrame(rows).to_csv(SD + 'SuppFig6_statistics.csv', index=False, float_format='%.3f')

# ---- Fig 6 and Supp Figs 7-9: Q-Q quantiles and tail underestimation (as cells/c25_fig6.py)
P = np.round(np.linspace(0, 1, 1001), 3)
qrows, trows = [], []
for n in NAMES:
    dm, dc = to_gC_month(model_results[n]), to_gC_month(cnn_results[n])
    mm_m = dm.sel(time=slice('2010-01-01', '2021-12-31')).groupby('time.month').mean('time')
    mm_c = dc.sel(time=slice('2010-01-01', '2021-12-31')).groupby('time.month').mean('time')
    xa = (dm.groupby('time.month') - mm_m).sel(time=slice('2022-01-01', '2023-12-31')).values.flatten()
    ya = (dc.groupby('time.month') - mm_c).sel(time=slice('2022-01-01', '2023-12-31')).values.flatten()
    ok = ~np.isnan(xa) & ~np.isnan(ya); x, y = xa[ok], ya[ok]
    qx, qy = np.quantile(x, P), np.quantile(y, P)
    qrows += [{'Model': disp(n), 'Probability': p, 'Model_anomaly_gC_m-2_month-1': a, 'Emulator_anomaly_gC_m-2_month-1': b}
              for p, a, b in zip(P, qx, qy)]
    rq01, eq01, rq99, eq99 = np.quantile(x, 0.01), np.quantile(y, 0.01), np.quantile(x, 0.99), np.quantile(y, 0.99)
    trows.append({'Model': disp(n), 'n_samples': int(ok.sum()), 'Model_q01': rq01, 'Emulator_q01': eq01,
                  'Underestimation_q01_pct': 100 * (1 - eq01 / rq01), 'Model_q99': rq99, 'Emulator_q99': eq99,
                  'Underestimation_q99_pct': 100 * (1 - eq99 / rq99)})
pd.DataFrame(qrows).to_csv(SD + 'Fig6_SuppFig7-9_QQ_quantiles.csv', index=False, float_format='%.4f')
pd.DataFrame(trows).to_csv(SD + 'Fig6_SuppFig7-9_tail_underestimation.csv', index=False, float_format='%.3f')

# ---- Supp Fig 2 and Supp Tables 3-6
shutil.copy(OUT_DIR + 'SuppFig2_values.csv', SD + 'SuppFig2_regional_values.csv')
for k in (3, 4, 5, 6):
    shutil.copy(OUT_DIR + f'SuppTable{k}.csv', SD + f'SuppTable{k}.csv')
print('source data written:', sorted(os.listdir(SD)))
