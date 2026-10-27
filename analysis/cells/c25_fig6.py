# Fig 6: Q-Q plots of model vs emulator monthly anomalies (2022-2023, each dataset minus its own 2010-2021
# climatology) for SDGVM, ORCHIDEE and EDv3; tail underestimation at the 1st and 99th percentiles of all 19 models in
# Fig6_tail_underestimation.txt.
apply_style(mathtext='regular')
unit_str = "(gC m$^{-2}$ month$^{-1}$)"

def qq_anomaly_pairs(model_name):
    """Test-period (2022-2023) anomalies in gC m-2 month-1 (actual days), each dataset minus its own
    2010-2021 monthly climatology."""
    dm = to_gC_month(model_results[model_name])
    dc = to_gC_month(cnn_results[model_name])
    mm_model = dm.sel(time=slice('2010-01-01', '2021-12-31')).groupby('time.month').mean('time')
    mm_cnn = dc.sel(time=slice('2010-01-01', '2021-12-31')).groupby('time.month').mean('time')
    model_anom = (dm.groupby('time.month') - mm_model).sel(time=slice('2022-01-01', '2023-12-31'))
    cnn_anom = (dc.groupby('time.month') - mm_cnn).sel(time=slice('2022-01-01', '2023-12-31'))
    x_raw = model_anom.values.flatten()
    y_raw = cnn_anom.values.flatten()
    mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
    return x_raw[mask], y_raw[mask]

def draw_qq(ax, model_name, label):
    x, y = qq_anomaly_pairs(model_name)
    n = min(len(x), len(y))
    qx = np.sort(x)[:n]
    qy = np.sort(y)[:n]
    ax.scatter(qx, qy, s=1.5, alpha=0.6, edgecolors='none', color='#1f77b4', rasterized=True)
    global_min = min(qx.min(), qy.min()); global_max = max(qx.max(), qy.max())
    margin = (global_max - global_min) * 0.05
    plot_min, plot_max = global_min - margin, global_max + margin
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=LW_REF, alpha=0.8, zorder=0)
    q01 = np.quantile(qx, 0.01); q99 = np.quantile(qx, 0.99)
    ax.axvline(q01, color='#d62728', linestyle='--', lw=LW_REF, alpha=0.8)
    ax.axvline(q99, color='#d62728', linestyle='--', lw=LW_REF, alpha=0.8)
    ax.set_xlim(plot_min, plot_max); ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.4)
    ax.set_title(disp(model_name), fontsize=FS, fontweight='bold', pad=3)
    panel_label(ax, label, x=-0.05, y=1.05)
    text_y_pos = plot_min + (plot_max - plot_min) * 0.1
    lab_bg = dict(facecolor='white', edgecolor='none', pad=0.3)   # keeps the dotted grid from running through the label
    ax.annotate('1%', xy=(q01, text_y_pos), xytext=(-2, 0), textcoords='offset points', color='#d62728',
                ha='right', va='center', fontsize=FS_SMALL, fontweight='bold', bbox=lab_bg)
    ax.annotate('99%', xy=(q99, text_y_pos), xytext=(2, 0), textcoords='offset points', color='#d62728',
                ha='left', va='center', fontsize=FS_SMALL, fontweight='bold', bbox=lab_bg)

selected_models = ['SDGVM', 'ORCHIDEE', 'EDv3']
panel_labels = ['a', 'b', 'c']
fig, axes = plt.subplots(1, 3, figsize=(FIG_W2, FIG_W2 * 5 / 15), constrained_layout=True)
for i, (ax, model_name) in enumerate(zip(axes, selected_models)):
    draw_qq(ax, model_name, panel_labels[i])
    ax.set_xlabel(f'Model anomaly\n{unit_str}')
    if i == 0:
        ax.set_ylabel(f'Emulator anomaly\n{unit_str}')
save_fig(fig, 'Fig6', fit_panels=True)
plt.show()

# ---- emulator underestimation at the reference 1st / 99th percentiles, all 19 models ----
rows = []
for model_name in DGVM_list.values():
    x, y = qq_anomaly_pairs(model_name)
    rq01, eq01 = np.quantile(x, 0.01), np.quantile(y, 0.01)
    rq99, eq99 = np.quantile(x, 0.99), np.quantile(y, 0.99)
    rows.append({'Model': disp(model_name), 'n_samples': len(x),
                 'ref_q01': rq01, 'emu_q01': eq01, 'underestimation_q01_pct': 100 * (1 - eq01 / rq01),
                 'ref_q99': rq99, 'emu_q99': eq99, 'underestimation_q99_pct': 100 * (1 - eq99 / rq99)})
tail_under = pd.DataFrame(rows).set_index('Model')
med = tail_under[['underestimation_q01_pct', 'underestimation_q99_pct']].median()
with open(OUT_DIR + 'Fig6_tail_underestimation.txt', 'w') as fh:
    fh.write("Fig 6: emulator underestimation of extreme monthly NBP anomalies (2022-2023, valid_mask_final, anomalies vs own 2010-2021 climatology, gC m-2 month-1)\n")
    fh.write("ref_qXX / emu_qXX = XX-th percentile of the pooled grid-cell-month anomalies of the reference model / of the emulator (the Q-Q pairing in Fig 6).\n")
    fh.write("underestimation_pct = 100 * (1 - emu_q / ref_q): positive = emulator percentile less extreme than the reference percentile.\n\n")
    fh.write(tail_under.to_string(float_format=lambda v: f"{v:.3f}"))
    fh.write(f"\n\nMedian of the 19 models: underestimation at 1st percentile = {med['underestimation_q01_pct']:.1f} %, at 99th percentile = {med['underestimation_q99_pct']:.1f} %\n")
print(tail_under.round(2).to_string())
print(med.round(2).to_string())
