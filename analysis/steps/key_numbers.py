# Key numbers quoted in the text (global, grid and tail statistics, 2022-2023, experiment 1). Writes OUT_DIR/key_numbers.txt.
# Runs after cells c16, c34, c108a.
def _pix_r2_mean(obs, pred, dim='time'):
    v = obs.std(dim) > 1e-10
    return float((xr.corr(obs, pred, dim=dim) ** 2).where(v).mean())

def _zkeep(da, k=10.0):
    arr = da.values
    f = np.isfinite(arr).all(0)
    clim = arr[:144].reshape(12, 12, *arr.shape[1:]).mean(0)
    an = arr - np.tile(clim, (14, 1, 1))
    sc = np.full(f.shape, np.nan); sc[f] = np.abs(an[:, f]).max(0)
    z = (sc - np.nanmean(sc[f])) / np.nanstd(sc[f])
    return xr.DataArray(~(z > k), coords={'lat': da.lat, 'lon': da.lon}, dims=('lat', 'lon')), int(np.nansum(z > k))

def key_numbers(MR, CR):
    names = list(DGVM_list.values()); out = {}
    G = {n: (cal_mon(MR[n]).values, cal_mon(CR[n]).values) for n in names}
    T = slice(144, 168)
    r2m = {n: pearson_r2(G[n][0][T], G[n][1][T]) for n in names}
    out['g_month_min'] = (min(r2m.values()), disp(min(r2m, key=r2m.get)))
    AN = {n: (own_anom(G[n][0]), own_anom(G[n][1])) for n in names}
    r2a = {n: pearson_r2(AN[n][0][T], AN[n][1][T]) for n in names}
    out['g_anom_median'] = float(np.median(list(r2a.values())))
    out['g_anom_min'] = (min(r2a.values()), disp(min(r2a, key=r2a.get)))
    out['g_anom_max'] = (max(r2a.values()), disp(max(r2a, key=r2a.get)))
    better = 0
    for n in names:
        am, ae = AN[n]
        reg = X_ERA5[:168] @ np.linalg.lstsq(X_ERA5[:144], am[:144], rcond=None)[0]
        better += int(_ss(ae[T], am[T], reg[T]) > 0)
    out['n_better_than_reg'] = better
    gm = np.mean([G[n][0] for n in names], 0); ge = np.mean([G[n][1] for n in names], 0)
    ym, ye = gm.reshape(14, 12).sum(1), ge.reshape(14, 12).sum(1)
    out['ens_dNBP_model'] = ym[13] - ym[12]
    out['ens_capture'] = (ye[13] - ye[12]) / (ym[13] - ym[12])
    out['ens_bias_2022'] = ye[12] - ym[12]; out['ens_bias_2023'] = ye[13] - ym[13]
    n22 = n23 = 0; tb = {}
    for n in names:
        b = G[n][1].reshape(14, 12).sum(1) - G[n][0].reshape(14, 12).sum(1)
        mu, sd = b[:12].mean(), b[:12].std(ddof=1)
        n22 += int(abs(b[12] - mu) > 2 * sd); n23 += int(abs(b[13] - mu) > 2 * sd); tb[n] = mu
    out['n_out_2022'] = n22; out['n_out_2023'] = n23
    out['tb_ISBA'] = tb['ISBA-CTRIP']; out['tb_CLASSIC'] = tb['CLASSIC']
    acc = {k: [] for k in ['pix_month', 'pix_clim', 'pix_anom', 'sc_train', 'sc_test_pix', 'sc_test_pool', 'sd_ratio',
                           'pool_22', 'pool_23', 'pair_lo', 'pair_hi', 'q01', 'q99', 'africa']}
    for n in names:
        dm, dc = to_gC_month(MR[n]), to_gC_month(CR[n])
        tm, tc = dm.sel(time=TEST_SLICE), dc.sel(time=TEST_SLICE)
        cm, cc = clim_ref(dm), clim_ref(dc)
        acc['pix_month'].append(_pix_r2_mean(tm, tc))
        base = xr.DataArray(cm.sel(month=tm['time.month']).values, coords=tm.coords, dims=tm.dims)
        acc['pix_clim'].append(_pix_r2_mean(tm, base))
        am_, ac_ = calculate_anomaly(tm, cm), calculate_anomaly(tc, cc)
        acc['pix_anom'].append(_pix_r2_mean(am_, ac_))
        acc['sc_train'].append(_pix_r2_mean(cm, cc, dim='month'))
        scm, scc = seasonal_cycle_test(dm), seasonal_cycle_test(dc)
        acc['sc_test_pix'].append(_pix_r2_mean(scm, scc, dim='month'))
        acc['sc_test_pool'].append(pearson_r2(scm.values, scc.values))
        x = am_.values.ravel(); y = ac_.values.ravel(); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        acc['sd_ratio'].append(float(np.std(y) / np.std(x)))
        acc['pool_22'].append(pearson_r2(am_.sel(time='2022').values, ac_.sel(time='2022').values))
        acc['pool_23'].append(pearson_r2(am_.sel(time='2023').values, ac_.sel(time='2023').values))
        q01, q99 = np.quantile(x, [0.01, 0.99]); lo, hi = x <= q01, x >= q99
        acc['pair_lo'].append(float(abs(y[lo].mean()) / abs(x[lo].mean()))); acc['pair_hi'].append(float(y[hi].mean() / x[hi].mean()))
        acc['q01'].append(float(abs(np.quantile(y, 0.01)) / abs(q01))); acc['q99'].append(float(np.quantile(y, 0.99) / q99))
        oa = own_anom(_regional_monthly_gtc(MR[n], region_masks['Africa']).values)
        ea = own_anom(_regional_monthly_gtc(CR[n], region_masks['Africa']).values)
        acc['africa'].append(pearson_r2(oa[T], ea[T]))
    for k, v in acc.items():
        out[k] = float(np.median(v))
    return out

KN = key_numbers(model_results, cnn_results)
LABELS = [
    ('g_month_min', 'Global monthly NBP R2, 2022-2023, lowest of 19 models'),
    ('g_anom_median', 'Global monthly anomaly R2, median of 19'),
    ('g_anom_min', 'Global monthly anomaly R2, lowest'),
    ('g_anom_max', 'Global monthly anomaly R2, highest'),
    ('n_better_than_reg', 'Number of models with SS > 0 vs ERA5 regression (of 19)'),
    ('ens_dNBP_model', '19-model mean: 2022->2023 change in model NBP (GtC yr-1)'),
    ('ens_capture', '19-model mean: fraction of that change captured by the emulators'),
    ('ens_bias_2022', '19-model mean annual bias 2022 (GtC yr-1)'),
    ('ens_bias_2023', '19-model mean annual bias 2023 (GtC yr-1)'),
    ('n_out_2022', 'Models with annual bias outside training mean +/- 2 sd, 2022'),
    ('n_out_2023', 'Models with annual bias outside training mean +/- 2 sd, 2023'),
    ('tb_ISBA', 'Training-period mean annual bias, ISBA-CTRIP (GtC yr-1)'),
    ('tb_CLASSIC', 'Training-period mean annual bias, CLASSIC (GtC yr-1)'),
    ('pix_month', 'Grid monthly NBP R2 (per cell, mean over cells), median of 19, emulator'),
    ('pix_clim', 'Same, climatology baseline (2010-2021 monthly climatology as prediction)'),
    ('pix_anom', 'Grid monthly anomaly R2 (per cell, mean over cells), median of 19'),
    ('sc_train', 'Seasonal cycle R2, training period 2010-2021 (per cell, mean), median'),
    ('sc_test_pix', 'Seasonal cycle R2, test period 2022-2023 (per cell, mean), median'),
    ('sc_test_pool', 'Seasonal cycle R2, test period 2022-2023 (pooled), median'),
    ('sd_ratio', 'Grid anomaly sd ratio emulator/model (pooled cell-months), median'),
    ('pair_lo', 'Tail, paired lower (mean of cells <= model 1st pct), |emulator|/|model|, median'),
    ('pair_hi', 'Tail, paired upper (mean of cells >= model 99th pct), emulator/model, median'),
    ('q01', 'Tail, quantile ratio at 1st percentile, median'),
    ('q99', 'Tail, quantile ratio at 99th percentile, median'),
    ('africa', 'Africa regional monthly anomaly R2 (2022-2023), median of 19'),
    ('pool_22', 'Pooled grid anomaly R2, 2022, median of 19'),
    ('pool_23', 'Pooled grid anomaly R2, 2023, median of 19'),
]
lines = []
for key, label in LABELS:
    v = KN[key]
    lines.append(f"{label}: {v[0]:.3f} ({v[1]})" if isinstance(v, tuple) else f"{label}: {v:.3f}")
with open(OUT_DIR + 'key_numbers.txt', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
