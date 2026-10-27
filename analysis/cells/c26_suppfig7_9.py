# Supp Figs 7-9: as Fig 6 for the other 16 models, grouped by tail underestimation.
import math
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

# Grouping: the 16 models other than SDGVM, ORCHIDEE and EDv3, sorted by the mean of underestimation_q01_pct and
# underestimation_q99_pct (Fig 6 table, tail_under), smallest first; 5 / 6 / 5 models
_inv = {disp(n): n for n in DGVM_list.values()}
_tm = tail_under[['underestimation_q01_pct', 'underestimation_q99_pct']].mean(axis=1).drop([disp(n) for n in ['SDGVM', 'ORCHIDEE', 'EDv3']]).sort_values()
_ord = [_inv[d] for d in _tm.index]
models_class1, models_class2, models_class3 = _ord[:5], _ord[5:11], _ord[11:]
print('Supp Figs 7-9 grouping (mean tail underestimation, %): ' + ' | '.join(', '.join(f'{disp(n)} {_tm[disp(n)]:.2f}' for n in g) for g in (models_class1, models_class2, models_class3)))
tasks = [
    {'name': 'SuppFig7', 'models': models_class1},
    {'name': 'SuppFig8', 'models': models_class2},
    {'name': 'SuppFig9', 'models': models_class3}
]
for task in tasks:
    model_list = task['models']
    ncols = 3
    nrows = math.ceil(len(model_list) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W2, FIG_W2 * 5 * nrows / 15), constrained_layout=True)
    axes_flat = axes.flatten()
    print(f"Drawing {task['name']} with {len(model_list)} models...")
    for i, ax in enumerate(axes_flat):
        if i >= len(model_list):
            ax.set_visible(False)   # empty slot: invisible, so it is not counted when the figure is cropped
            continue
        draw_qq(ax, model_list[i], chr(97 + i))
        ax.set_xlabel(f'Model anomaly\n{unit_str}')
        if i % ncols == 0:
            ax.set_ylabel(f'Emulator anomaly\n{unit_str}')
    save_fig(fig, task['name'], fit_panels=True)
    plt.show()
    plt.close()
