# Experiment 1, one-step vs recursive prediction, and the grid seasonal-cycle climatology baseline. Writes to OUT_DIR/checks/:
# 1. identity_recursive_vs_onestep_2010_2021.csv: the 2010-2021 rows of the recursive files equal the one-step files
#    (they are copied by --historical_result_dir);
# 2. C4_onestep_vs_recursive_by_model.csv: pooled grid anomaly R2, global monthly anomaly R2 and annual bias, recursive vs
#    one-step, all 19 models, and the 2021 (training-period) pooled grid anomaly R2;
# 3. grid_seasonal_cycle_climatology_R2.csv: per-cell R2 of the 2022-2023 mean seasonal cycle, emulator vs the model's
#    own 2010-2021 climatology used as prediction.
# Runs after cells c16, c107a, c108a and c24a.
import os
OUT = OUT_DIR + 'checks/'; os.makedirs(OUT, exist_ok=True)
NAMES = list(DGVM_list.values()); ID = {v: k for k, v in DGVM_list.items()}
ONE = EMULATOR_EXP1_ONESTEP_DIR
def load_pred(path, like, mask):
    df = pd.read_csv(path, engine='pyarrow', usecols=['year', 'month', 'predictions'])
    df = df[(df['year'] >= 2010) & (df['year'] < 2024)]
    a = df['predictions'].values / 1e10
    assert a.size == 168 * 259200, (path, a.size)
    return xr.DataArray(a.reshape(168, 360, 720), coords=like.coords, dims=like.dims).where(mask)
def pooled_by_year(dm_raw, dc_raw, years):
    dm, dc = to_gC_month(dm_raw), to_gC_month(dc_raw)
    am, ac = calculate_anomaly(dm, clim_ref(dm)), calculate_anomaly(dc, clim_ref(dc))
    return {y: pearson_r2(am.sel(time=y).values, ac.sel(time=y).values) for y in years}
def glob_stats(dm_raw, dc_raw):
    gm, ge = cal_mon(dm_raw).values, cal_mon(dc_raw).values
    am, ae = own_anom(gm), own_anom(ge)
    b = ge.reshape(14, 12).sum(1) - gm.reshape(14, 12).sum(1)
    return {'GlobalMonthlyAnomR2_2022': pearson_r2(am[144:156], ae[144:156]), 'GlobalMonthlyAnomR2_2023': pearson_r2(am[156:168], ae[156:168]),
            'GlobalMonthlyAnomR2_2022_2023': pearson_r2(am[144:168], ae[144:168]), 'GlobalMonthlyNBP_R2_2022_2023': pearson_r2(gm[144:168], ge[144:168]),
            'AnnualBias_GtC_yr_2022': b[12], 'AnnualBias_GtC_yr_2023': b[13], 'MeanBias_GtC_yr_2010_2021': b[:12].mean(), 'SDBias_GtC_yr_2010_2021': b[:12].std(ddof=1)}
idrows, c4rows, scrows = [], [], []
for n in NAMES:
    mr, cr, fm = model_results[n], cnn_results[n], final_masks[n]
    one = load_pred(ONE + f'{ID[n]}.csv', cr, fm)
    d = np.abs(cr.values[:144] - one.values[:144]); d = d[np.isfinite(d)]
    nanmis = int((np.isnan(cr.values[:144]) != np.isnan(one.values[:144])).sum())
    idrows.append({'Model': disp(n), 'max_abs_diff_2010_2021_kg_m2_s1': float(d.max()) if d.size else np.nan, 'NaN_mismatch': nanmis})
    pr, po = pooled_by_year(mr, cr, ('2021', '2022', '2023')), pooled_by_year(mr, one, ('2021', '2022', '2023'))
    gr, go = glob_stats(mr, cr), glob_stats(mr, one)
    c4rows.append({'Model': disp(n),
                   'PooledGridAnomR2_2021_insample': pr['2021'],
                   'PooledGridAnomR2_2022_recursive': pr['2022'], 'PooledGridAnomR2_2022_onestep': po['2022'],
                   'PooledGridAnomR2_2023_recursive': pr['2023'], 'PooledGridAnomR2_2023_onestep': po['2023'],
                   'GlobalMonthlyAnomR2_2022_recursive': gr['GlobalMonthlyAnomR2_2022'], 'GlobalMonthlyAnomR2_2022_onestep': go['GlobalMonthlyAnomR2_2022'],
                   'GlobalMonthlyAnomR2_2023_recursive': gr['GlobalMonthlyAnomR2_2023'], 'GlobalMonthlyAnomR2_2023_onestep': go['GlobalMonthlyAnomR2_2023'],
                   'AnnualBias_GtC_yr_2022_recursive': gr['AnnualBias_GtC_yr_2022'], 'AnnualBias_GtC_yr_2022_onestep': go['AnnualBias_GtC_yr_2022'],
                   'AnnualBias_GtC_yr_2023_recursive': gr['AnnualBias_GtC_yr_2023'], 'AnnualBias_GtC_yr_2023_onestep': go['AnnualBias_GtC_yr_2023']})
    del one
    dm, dc = to_gC_month(mr), to_gC_month(cr)
    scm = seasonal_cycle_test(dm)
    scrows.append({'Model': disp(n),
                   'SC_R2_emulator_test_vs_model_test': calc_pixel_wise_metrics(scm, seasonal_cycle_test(dc), time_dim='month')['r2_mean'],
                   'SC_R2_climatology_2010_2021_vs_model_test': calc_pixel_wise_metrics(scm, clim_ref(dm), time_dim='month')['r2_mean']})
    del dm, dc
    print('done', n, flush=True)
IDT = pd.DataFrame(idrows).set_index('Model'); IDT.to_csv(OUT + 'identity_recursive_vs_onestep_2010_2021.csv')
C4 = pd.DataFrame(c4rows).set_index('Model')
C4.loc['Median of 19'] = C4.median()
C4.to_csv(OUT + 'C4_onestep_vs_recursive_by_model.csv', float_format='%.4f')
SC = pd.DataFrame(scrows).set_index('Model'); SC.loc['Median of 19'] = SC.median(); SC.to_csv(OUT + 'grid_seasonal_cycle_climatology_R2.csv', float_format='%.4f')
print(IDT.to_string()); print(C4.round(3).to_string()); print(SC.round(3).to_string())
