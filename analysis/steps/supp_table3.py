# Supp Table 3: 3 x 3 summary (global / regional / grid cell x annual / monthly / monthly anomaly), emulator vs climatology
# baseline, from the outputs of the other jobs. Medians over the 19 models; regional = range over the ten regions. The
# grid monthly climatology R2 (0.76) is pix_clim in key_numbers.txt.
import pandas as pd, numpy as np
V, OUT = OUT_DIR, OUT_DIR + 'SuppTable3.csv'
R10 = ['North America', 'Europe', 'West Asia', 'Russia', 'East Asia', 'South Asia', 'Southeast Asia', 'Latin America', 'Oceania', 'Africa']
p41 = pd.read_csv(V + 'fit_for_purpose/P4_1_grid_mean_annual_map.csv', index_col=0); p41 = p41[[not str(i).lower().startswith('median') for i in p41.index]]
p42 = pd.read_csv(V + 'fit_for_purpose/P4_2_monthly_total_median.csv', index_col=0)
p43 = pd.read_csv(V + 'fit_for_purpose/P4_3_annual_error_summary.csv')
p43l = pd.read_csv(V + 'fit_for_purpose/P4_3_annual_error_long.csv')
p44 = pd.read_csv(V + 'fit_for_purpose/P4_4_anomaly_skill_vs_zero_median.csv', index_col=0)
p3 = pd.read_csv(V + 'fit_for_purpose/P3_regional_summary.csv')
fs = pd.read_csv(V + 'model_differences/Part1_Fig3_column_summary.csv', index_col=0)['median_19']
t3 = pd.read_csv(V + 'SuppTable5.csv', index_col=0).drop(index='19-model mean')
sc = pd.read_csv(V + 'checks/grid_seasonal_cycle_climatology_R2.csv', index_col=0).loc['Median of 19']
g = p43[p43.Region == 'Global'].set_index('Year'); r = p43[p43.Region.isin(R10)]
pix_clim = float([l for l in open(V + 'key_numbers.txt') if l.startswith('Same, climatology baseline')][0].rsplit(': ', 1)[1])
rr = lambda s: f'{s.min():.2f}-{s.max():.2f}'
def share(y):
    q = p43l[(p43l.Year == y) & p43l.Region.isin(R10)]; return f'{100 * (q.AbsErr_emulator < q.AbsErr_climatology).mean():.0f}%'
ra = p3[(p3.metric == 'MonthlyAnomaly_R2') & p3.Region.isin(R10)]['median_19']
rows = [
 ('Global', 'Annual', 'Median |annual NBP error| (GtC yr-1), 2022 / 2023', f"{g.loc[2022, 'median_AbsErr_emulator']:.2f} / {g.loc[2023, 'median_AbsErr_emulator']:.2f}", f"{g.loc[2022, 'median_AbsErr_climatology']:.2f} / {g.loc[2023, 'median_AbsErr_climatology']:.2f}",
  f"models with |error| < 0.4: emulator {g.loc[2022, 'n_AbsErr_emulator_lt_0.4']} / {g.loc[2023, 'n_AbsErr_emulator_lt_0.4']}, climatology {g.loc[2022, 'n_AbsErr_climatology_lt_0.4']} / {g.loc[2023, 'n_AbsErr_climatology_lt_0.4']}; emulator error smaller for {g.loc[2022, 'n_emulator_better']} / {g.loc[2023, 'n_emulator_better']} of 19 models"),
 ('Global', 'Monthly', 'Median R2 of monthly NBP (RMSE, GtC month-1), 24 months', f"{p42.loc['Global', 'R2_emulator']:.3f} ({p42.loc['Global', 'RMSE_emulator']:.3f})", f"{p42.loc['Global', 'R2_climatology']:.3f} ({p42.loc['Global', 'RMSE_climatology']:.3f})",
  f"emulator R2 higher for {p42.loc['Global', 'n_models_emulator_R2_gt_climatology_R2']} of 19 models"),
 ('Global', 'Monthly anomaly', 'Median R2 of monthly anomalies; skill score vs climatology (zero anomaly)', f"R2 {t3['R2_global_monthly_anomaly_2022_2023'].median():.2f}; SS {p44.loc['Global', 'median']:.2f}", 'R2 undefined (zero anomaly); SS = 0 by definition',
  f"SS > 0 for {p44.loc['Global', 'n_models_SS_gt_0']} of 19 models"),
 ('Regional (10 RECCAP2 regions)', 'Annual', 'Median |annual NBP error| (GtC yr-1), range over regions and years', rr(r['median_AbsErr_emulator']), rr(r['median_AbsErr_climatology']),
  f"emulator error smaller in {share(2022)} (2022) and {share(2023)} (2023) of 190 model-region cases"),
 ('Regional (10 RECCAP2 regions)', 'Monthly', 'Median R2 of monthly NBP, range over regions', rr(p42.loc[R10, 'R2_emulator']), rr(p42.loc[R10, 'R2_climatology']),
  f"emulator median higher in {int((p42.loc[R10, 'R2_emulator'] > p42.loc[R10, 'R2_climatology']).sum())} of 10 regions"),
 ('Regional (10 RECCAP2 regions)', 'Monthly anomaly', 'Median R2 of monthly anomalies; median skill score vs climatology, ranges over regions', f"R2 {rr(ra)}; SS {rr(p44.loc[R10, 'median'])}", 'R2 undefined; SS = 0',
  f"median SS > 0.5 in {int((p44.loc[R10, 'median'] > 0.5).sum())} of 10 regions"),
 ('Grid cell', 'Annual', 'Spatial R2 of the 2022-2023 mean annual NBP map (RMSE, gC m-2 yr-1)', f"{p41['R2_emulator_map_vs_2022_2023'].median():.2f} ({p41['RMSE_emulator_gC_m2_yr'].median():.1f})", f"{p41['R2_climatology_map_vs_2022_2023'].median():.2f} ({p41['RMSE_climatology_gC_m2_yr'].median():.1f})",
  f"climatology = 2010-2021 mean annual map; emulator R2 higher for {int((p41['R2_emulator_map_vs_2022_2023'] > p41['R2_climatology_map_vs_2022_2023']).sum())} and RMSE lower for {int((p41['RMSE_emulator_gC_m2_yr'] < p41['RMSE_climatology_gC_m2_yr']).sum())} of 19 models"),
 ('Grid cell', 'Monthly', 'Mean per-cell R2 of monthly NBP (24 months); mean seasonal cycle R2', f"{fs['R2_Monthly NBP']:.2f}; seasonal cycle {sc['SC_R2_emulator_test_vs_model_test']:.2f}", f"{pix_clim:.2f}; seasonal cycle {sc['SC_R2_climatology_2010_2021_vs_model_test']:.2f}",
  'climatology = 2010-2021 monthly climatology of the model; seasonal cycle: 2010-2021 mean seasonal cycle vs the 2022-2023 one'),
 ('Grid cell', 'Monthly anomaly', 'Mean per-cell R2 of monthly anomalies; skill score vs climatology (all cells and months pooled)', f"R2 {fs['R2_Monthly anomalies']:.2f}; SS {p44.loc['Grid cells (pooled)', 'median']:.2f}", 'R2 undefined; SS = 0',
  f"SS > 0 for {p44.loc['Grid cells (pooled)', 'n_models_SS_gt_0']} of 19 models"),
]
T = pd.DataFrame(rows, columns=['Scale', 'Resolution', 'Metric (median over 19 models)', 'Emulator', 'Climatology baseline', 'Note'])
T.to_csv(OUT, index=False); print(T.to_string())
