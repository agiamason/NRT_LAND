# Supp Table 6 sensitivity: 2024 annual anomaly of the version-consistent subset for R2 thresholds 0.85-0.95 (paper
# rule: 0.88). Runs after cell c109. Writes to OUT_DIR/checks/.
import os
OUT = OUT_DIR + 'checks/'
os.makedirs(OUT, exist_ok=True)
R2V = {n: r['R2_v14_vs_v13_monthly_anomaly_2010_2021'] for n, r in zip(ENS, rows5)}
rows = []
for th in (0.85, 0.88, 0.90, 0.95):
    sub = [n for n in ENS if R2V[n] > th]
    r = row5(f'R2 > {th}', {k: np.mean([S5[n][k] for n in sub], 0) for k in ['g13', 'g14', 'gA']})
    r.update({'threshold': th, 'n_models': len(sub), 'models': ', '.join(disp(n) for n in sub)}); rows.append(r)
r = row5('all 17', {k: np.mean([S5[n][k] for n in ENS], 0) for k in ['g13', 'g14', 'gA']}); r.update({'threshold': np.nan, 'n_models': len(ENS), 'models': 'all'}); rows.append(r)
TH = pd.DataFrame(rows).set_index('Model'); TH.to_csv(OUT + 'C2_2024_subset_threshold_sensitivity.csv', float_format='%.4f')
pd.Series(R2V).rename(index=disp).to_csv(OUT + 'C2_R2_v14_vs_v13_by_model.csv', header=['R2_v14_vs_v13_monthly_anomaly_2010_2021'], float_format='%.4f')
print(TH.drop(columns=['models']).round(3).to_string()); print(TH['models'].to_string())
