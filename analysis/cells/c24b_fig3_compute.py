# Fig 3, part 2: per-model R2 and RMSE for five quantities in 2022-2023 (monthly NBP, mean annual NBP, mean seasonal
# cycle, monthly anomalies, quarterly anomalies), per-cell metrics averaged over cells, plus the ensemble median.
# ================= 1. Metrics: 19 models + ensemble median =================
all_models = sorted(list(model_results.keys()))
print("Calculating Ensemble median...")
list_obs = [model_results[m].sel(time=FULL_SLICE).drop_vars('DGVM', errors='ignore') for m in all_models]
list_emu = [cnn_results[m].sel(time=FULL_SLICE).drop_vars('DGVM', errors='ignore') for m in all_models]
obs_median = xr.concat(list_obs, dim='model').median(dim='model')
emu_median = xr.concat(list_emu, dim='model').median(dim='model')
del list_obs, list_emu
current_model_results = model_results.copy()
current_cnn_results = cnn_results.copy()
current_model_results['Ensemble median'] = obs_median
current_cnn_results['Ensemble median'] = emu_median
models_to_process = all_models + ['Ensemble median']

col_names = ['Monthly NBP', 'Mean annual NBP', 'Mean seasonal cycle', 'Monthly anomalies', 'Quarterly anomalies']
metrics_data, scale_rows = [], []
for name in models_to_process:
    raw_m, raw_c = current_model_results[name], current_cnn_results[name]
    dm, dc = to_gC_month(raw_m), to_gC_month(raw_c)
    # [Col 1] Monthly NBP
    m1 = calc_pixel_wise_metrics(dm.sel(time=TEST_SLICE), dc.sel(time=TEST_SLICE))
    # [Col 2] Mean annual NBP (spatial)
    s2 = spatial_metrics(annual_gC(raw_m.sel(time=TEST_SLICE)).mean('year'), annual_gC(raw_c.sel(time=TEST_SLICE)).mean('year'))
    # [Col 3] Mean seasonal cycle, 2022-2023 test period
    sc_model = dm.sel(time=TEST_SLICE).groupby('time.month').mean('time')
    sc_cnn = dc.sel(time=TEST_SLICE).groupby('time.month').mean('time')
    m3 = calc_pixel_wise_metrics(sc_model, sc_cnn, time_dim='month')
    # [Col 4] Monthly anomalies (own 2010-2021 climatology)
    clim_model = dm.sel(time=REF_SLICE).groupby('time.month').mean('time')
    clim_cnn = dc.sel(time=REF_SLICE).groupby('time.month').mean('time')
    m4 = calc_pixel_wise_metrics(calculate_anomaly(dm.sel(time=TEST_SLICE), clim_model),
                                 calculate_anomaly(dc.sel(time=TEST_SLICE), clim_cnn))
    # [Col 5] Quarterly anomalies
    dm_q = dm.resample(time='QS-DEC').mean()
    dc_q = dc.resample(time='QS-DEC').mean()
    clim_model_q = dm_q.sel(time=REF_SLICE).groupby('time.season').mean('time')
    clim_cnn_q = dc_q.sel(time=REF_SLICE).groupby('time.season').mean('time')
    m5 = calc_pixel_wise_metrics(dm_q.sel(time=TEST_SLICE).groupby('time.season') - clim_model_q,
                                 dc_q.sel(time=TEST_SLICE).groupby('time.season') - clim_cnn_q)
    metrics_data.append([m1['r2_mean'], s2['r2'], m3['r2_mean'], m4['r2_mean'], m5['r2_mean'],
                         m1['rmse_median'], s2['rmse'], m3['rmse_median'], m4['rmse_median'], m5['rmse_median']])
    # global scale (global totals, GtC month-1)
    gm, gc = cal_mon(raw_m), cal_mon(raw_c)
    gmt, gct = gm.sel(time=TEST_SLICE), gc.sel(time=TEST_SLICE)
    gcm, gcc = gm.sel(time=REF_SLICE).groupby('time.month').mean('time'), gc.sel(time=REF_SLICE).groupby('time.month').mean('time')
    gm_q, gc_q = gm.resample(time='QS-DEC').mean(), gc.resample(time='QS-DEC').mean()
    gqa_m = gm_q.sel(time=TEST_SLICE).groupby('time.season') - gm_q.sel(time=REF_SLICE).groupby('time.season').mean('time')
    gqa_c = gc_q.sel(time=TEST_SLICE).groupby('time.season') - gc_q.sel(time=REF_SLICE).groupby('time.season').mean('time')
    scale_rows.append({
        'Model': disp(name),
        'MonthlyNBP_grid_median': m1['r2_median'], 'MonthlyNBP_grid_pooled': m1['r2_pooled'],
        'MonthlyNBP_global(n=24)': pearson_r2(gmt, gct),
        'MeanAnnual_grid_spatial': s2['r2'], 'MeanAnnual_global': np.nan,
        'SeasonalCycle_grid_median': m3['r2_median'], 'SeasonalCycle_grid_pooled': m3['r2_pooled'],
        'SeasonalCycle_global(n=12)': pearson_r2(gmt.groupby('time.month').mean('time'), gct.groupby('time.month').mean('time')),
        'MonthlyAnom_grid_median': m4['r2_median'], 'MonthlyAnom_grid_pooled': m4['r2_pooled'],
        'MonthlyAnom_global(n=24)': pearson_r2(calculate_anomaly(gmt, gcm), calculate_anomaly(gct, gcc)),
        'QuarterlyAnom_grid_median': m5['r2_median'], 'QuarterlyAnom_grid_pooled': m5['r2_pooled'],
        'QuarterlyAnom_global(n=8)': pearson_r2(gqa_m, gqa_c)})

df = pd.DataFrame(metrics_data, index=models_to_process,
                  columns=[f"R2_{c}" for c in col_names] + [f"RMSE_{c}" for c in col_names])

# ================= Row order =================
median_row = df.loc[['Ensemble median']]
models_only = df.drop(index='Ensemble median')
models_sorted = models_only.sort_values(by='R2_Mean annual NBP', ascending=True)
df_final = pd.concat([models_sorted, median_row])
df_final.index = [disp(n) for n in df_final.index]
df_r2 = df_final.iloc[:, 0:5].copy();  df_r2.columns = col_names
df_rmse = df_final.iloc[:, 5:10].copy(); df_rmse.columns = col_names

scale_df = pd.DataFrame(scale_rows).set_index('Model').loc[df_final.index]
