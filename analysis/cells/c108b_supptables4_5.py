# Supp Tables 4 (bias of global annual NBP) and 5 (R2, RMSE and skill scores of global monthly NBP), 2022-2023.
GLOB = {n: (cal_mon(model_results[n]).values, cal_mon(cnn_results[n]).values) for n in DGVM_list.values()}
GLOB['19-model mean'] = (np.mean([GLOB[n][0] for n in DGVM_list.values()], 0), np.mean([GLOB[n][1] for n in DGVM_list.values()], 0))
T = slice(144, 168)
rows3, rows4 = [], []
for name, (gm, ge) in GLOB.items():
    am, ae = own_anom(gm), own_anom(ge)
    beta = np.linalg.lstsq(X_ERA5[:144], am[:144], rcond=None)[0]
    reg = X_ERA5[:168] @ beta
    pa, pb = np.full(24, am[143]), np.full(24, am[132:144].mean())
    rows3.append({'Model': disp(name),
                  'R2_global_monthly_NBP_2022_2023': pearson_r2(gm[T], ge[T]),
                  'R2_global_monthly_anomaly_2022_2023': pearson_r2(am[T], ae[T]),
                  'RMSE_anomaly_2022_GtC_month-1': _rmse(ae[144:156], am[144:156]),
                  'RMSE_anomaly_2023_GtC_month-1': _rmse(ae[156:168], am[156:168]),
                  'SS_climatology': _ss(ae[T], am[T], np.zeros(24)),
                  'SS_persistence_a_Dec2021': _ss(ae[T], am[T], pa),
                  'SS_persistence_b_mean2021': _ss(ae[T], am[T], pb),
                  'SS_ERA5_regression': _ss(ae[T], am[T], reg[T]),
                  'Regression_fit_R2_2010_2021': pearson_r2(am[:144], reg[:144]),
                  'Regression_test_R2_2022_2023': pearson_r2(am[T], reg[T])})
    ym, ye = gm.reshape(14, 12).sum(1), ge.reshape(14, 12).sum(1)
    b = ye - ym; mu, sd = b[:12].mean(), b[:12].std(ddof=1)
    rows4.append({'Model': disp(name),
                  'Bias_mean_2010_2021_GtC_yr-1': mu, 'Bias_sd_2010_2021_GtC_yr-1': sd,
                  'Bias_2022_GtC_yr-1': b[12], 'Outside_mean_pm_2sd_2022': bool(abs(b[12] - mu) > 2 * sd),
                  'Bias_2023_GtC_yr-1': b[13], 'Outside_mean_pm_2sd_2023': bool(abs(b[13] - mu) > 2 * sd),
                  'dNBP_2023_minus_2022_model_GtC_yr-1': ym[13] - ym[12],
                  'dNBP_2023_minus_2022_emulator_GtC_yr-1': ye[13] - ye[12]})
supp3 = pd.DataFrame(rows3).set_index('Model'); supp4 = pd.DataFrame(rows4).set_index('Model')
supp3.to_csv(OUT_DIR + 'SuppTable5.csv', float_format='%.4f')
supp4.to_csv(OUT_DIR + 'SuppTable4.csv', float_format='%.4f')
print(supp3.round(3).to_string()); print(supp4.round(3).to_string())
