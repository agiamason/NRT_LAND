# Supp Fig 6 (89 mm): model vs emulator seasonal anomalies, against each dataset's own 2010-2021 climatology; axis limits
# symmetric about zero.
apply_style()

def get_truncated_purples():
    cmap = plt.get_cmap('Purples')
    minval = 0.5
    maxval = 1.0
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, 256)))
    return new_cmap

strict_purple_cmap = get_truncated_purples()

selected_models = ['ORCHIDEE', 'CLM6.0']
col_titles = ['JFM anomalies', 'JAS anomalies']
unit_mon = "(gC m$^{-2}$ month$^{-1}$)"
ref_slice = REF_SLICE
test_slice = TEST_SLICE

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(FIG_W1, FIG_W1 * 9.5 / 10), constrained_layout=True)
col_lims = {j: {'min': np.inf, 'max': -np.inf} for j in range(2)}
hb_list = []

for i, dgvm_name in enumerate(selected_models):
    da_model = to_gC_month(model_results[dgvm_name])
    da_cnn = to_gC_month(cnn_results[dgvm_name])
    # seasonal-mean anomaly per grid cell, as in Fig 5 (cell 30): monthly anomalies against each dataset's own
    # 2010-2021 climatology (clim_ref), averaged over the JFM or JAS months of 2022-2023
    anom_model_full = calculate_anomaly(da_model.sel(time=test_slice), clim_ref(da_model))
    anom_cnn_full = calculate_anomaly(da_cnn.sel(time=test_slice), clim_ref(da_cnn))
    mask_jfm = anom_model_full.time.dt.month.isin([1, 2, 3])
    x1 = anom_model_full.where(mask_jfm).mean('time').values.flatten()
    y1 = anom_cnn_full.where(mask_jfm).mean('time').values.flatten()
    mask_jas = anom_model_full.time.dt.month.isin([7, 8, 9])
    x2 = anom_model_full.where(mask_jas).mean('time').values.flatten()
    y2 = anom_cnn_full.where(mask_jas).mean('time').values.flatten()
    data_pairs = [(x1, y1), (x2, y2)]
    for j, (x_raw, y_raw) in enumerate(data_pairs):
        ax = axes[i, j]
        mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x, y = x_raw[mask], y_raw[mask]
        l_min, l_max = min(x.min(), y.min()), max(x.max(), y.max())
        if l_min < col_lims[j]['min']: col_lims[j]['min'] = l_min
        if l_max > col_lims[j]['max']: col_lims[j]['max'] = l_max
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap=strict_purple_cmap, mincnt=1, linewidths=0.2)
        if i == 0 and j == 0: hb_list.append(hb)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
        r2 = r_value**2
        x_fit = np.linspace(l_min, l_max, 100)
        ax.plot(x_fit, slope * x_fit + intercept, color='#d62728', lw=LW_LINE, zorder=3)
        stats_text = (f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nSlope = {slope:.2f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=FS_SMALL, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.3', linewidth=LW_AX))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for j in range(2):
    g_abs = max(abs(col_lims[j]['min']), abs(col_lims[j]['max']))
    lim_min, lim_max = -1.05 * g_abs, 1.05 * g_abs
    for i in range(2):
        ax = axes[i, j]
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=LW_REF, alpha=0.6, zorder=0)
        if i == 0:
            ax.set_title(col_titles[j], fontsize=FS, fontweight='bold', pad=4)
        if j == 0:
            ax.set_ylabel(f"Emulator {unit_mon}", fontsize=FS_SMALL)
            ax.annotate(selected_models[i], xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-3, 0),
                        textcoords='offset points', fontsize=FS, fontweight='bold', rotation=90, va='center', ha='right')
        else:
            ax.set_ylabel("")
        if i == 1:
            ax.set_xlabel(f"Model {unit_mon}", fontsize=FS_SMALL)

fig.align_ylabels(axes[:, 0])
cbar = fig.colorbar(hb_list[0], ax=axes, orientation='horizontal', fraction=0.03, pad=0.06, aspect=25, shrink=0.6)
cbar.set_label('Data density (log scale)', fontsize=FS)
cbar.ax.tick_params(labelsize=FS_TICK, width=LW_AX)
cbar.outline.set_linewidth(LW_AX)
save_fig(fig, 'SuppFig6', width_mm=89, fit_panels=True)
plt.show()
