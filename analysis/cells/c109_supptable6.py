# Supp Table 6: 2024 one-year-ahead test (experiment 2: emulators trained on 2000-2023). Truth = TRENDY v14 S2.
# Mask = v13 evaluation mask and v14 valid (complete 2010-2024, sd 2010-2023 > 1e-10) and finite emulator output.
# 17 models; version-consistent subset = models whose v14 and v13 global monthly anomalies (2010-2021) have R2 > 0.88.
ENS = ['ORCHIDEE', 'CABLE-POP', 'CLASSIC', 'IBIS', 'JSBACH', 'JULES', 'LPX-Bern', 'VISIT', 'iMAPLE', 'CLM6.0', 'LPJwsl', 'EDv3', 'ISAM', 'E3SM', 'CARDAMOM', 'SDGVM', 'visit-ut']
R2_SUBSET = 0.88   # subset rule: R2 between v14 and v13 global monthly anomalies, 2010-2021
DIR_A = EMULATOR_EXP2_DIR   # experiment 2: one-step rows 2010-2023, recursive 2024
_ID = {v: k for k, v in DGVM_list.items()}
_area = seaflux.area.area_grid(model_results['ORCHIDEE'].lat.values, model_results['ORCHIDEE'].lon.values, return_dataarray=True).values
S5 = {}
for name in ENS:
    dfa = pd.read_csv(DIR_A + f'{_ID[name]}.csv', engine='pyarrow', usecols=['year', 'month', 'predictions'])
    assert len(dfa) >= 180 * 259200 and dfa['year'].iloc[0] == 2010 and dfa['month'].iloc[0] == 1
    assert dfa['year'].iloc[180 * 259200 - 1] == 2024 and dfa['month'].iloc[180 * 259200 - 1] == 12
    pa_ = dfa['predictions'].values[:180 * 259200].reshape(180, 360, 720) / 1e10
    del dfa
    a14 = _v14.sel(DGVM=V14_NAME[name]).values.astype(np.float64)
    f14 = np.isfinite(a14).all(0); sd14 = np.full(f14.shape, np.nan); sd14[f14] = a14[:168][:, f14].std(0)
    m = final_masks[name].values & f14 & (sd14 > 1e-10) & np.isfinite(pa_).all(0)
    a13 = _v13.sel(DGVM=name).values.astype(np.float64)
    def _g(arr, n):
        return (arr[:n][:, m] * _area[m]).sum(1) * _sec180[:n] / 1e12
    S5[name] = {'g13': _g(a13, 168), 'g14': _g(a14, 180), 'gA': _g(pa_, 180), 'n': int(m.sum())}
    del pa_, a13, a14

def row5(label, d):
    a13, a14, aA = own_anom(d['g13']), own_anom(d['g14']), own_anom(d['gA'])
    y14, yA = d['g14'].reshape(15, 12).sum(1), d['gA'].reshape(15, 12).sum(1)
    an14, anA = y14[14] - y14[:12].mean(), yA[14] - yA[:12].mean()
    x, y = a14[168:180], aA[168:180]
    beta = np.linalg.lstsq(X_ERA5[:168], a13, rcond=None)[0]
    reg = X_ERA5[168:180] @ beta
    pers = np.full(12, a13[156:168].mean())
    return {'Model': label, 'n_cells': d.get('n', np.nan),
            'R2_v14_vs_v13_monthly_anomaly_2010_2021': pearson_r2(a13[:144], a14[:144]),
            'Annual_anomaly_2024_model_v14_GtC_yr-1': an14, 'Annual_anomaly_2024_emulator_GtC_yr-1': anA,
            'Same_sign_2024': bool(np.sign(an14) == np.sign(anA)),
            'R2_monthly_anomaly_2024': pearson_r2(x, y), 'RMSE_monthly_anomaly_2024_GtC_month-1': _rmse(y, x),
            'SS_climatology': _ss(y, x, np.zeros(12)), 'SS_persistence_v13_2023_mean': _ss(y, x, pers), 'SS_ERA5_regression': _ss(y, x, reg)}

rows5 = [row5(disp(n), S5[n]) for n in ENS]
SUB = [n for n, r in zip(ENS, rows5) if r['R2_v14_vs_v13_monthly_anomaly_2010_2021'] > R2_SUBSET]
ALL_LABEL, SUB_LABEL = f'{len(ENS)}-model mean', f'{len(SUB)}-model version-consistent subset mean'
rows5.append(row5(ALL_LABEL, {k: np.mean([S5[n][k] for n in ENS], 0) for k in ['g13', 'g14', 'gA']}))
rows5.append(row5(SUB_LABEL, {k: np.mean([S5[n][k] for n in SUB], 0) for k in ['g13', 'g14', 'gA']}))
print('Subset (R2 v14 vs v13 >', R2_SUBSET, '):', ', '.join(disp(n) for n in SUB))
supp5 = pd.DataFrame(rows5).set_index('Model')
supp5.to_csv(OUT_DIR + 'SuppTable6.csv', float_format='%.4f')
print(supp5.round(3).to_string())
