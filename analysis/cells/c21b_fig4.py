# Fig 4, part 2: grid-cell scatter of model vs emulator for ORCHIDEE and CLM6.0 (monthly NBP, mean annual NBP, mean
# seasonal cycle of the test period, monthly anomalies against each dataset's own 2010-2021 climatology).
selected_models = ['ORCHIDEE', 'CLM6.0']
col_titles = ['Monthly NBP', 'Mean annual NBP', 'Mean seasonal cycle', 'Monthly anomalies']
unit_mon = "(gC m$^{-2}$ month$^{-1}$)"
unit_yr = "(gC m$^{-2}$ yr$^{-1}$)"
ref_slice = REF_SLICE     # anomaly base period
test_slice = TEST_SLICE   # test period

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(FIG_W2, FIG_W2 * 9 / 16), constrained_layout=True)
col_lims = {j: {'min': np.inf, 'max': -np.inf} for j in range(4)}
hb_list = []

for i, dgvm_name in enumerate(selected_models):
    data_pairs = scatter_component_pairs(dgvm_name)
    for j, (x_raw, y_raw) in enumerate(data_pairs):
        ax = axes[i, j]
        mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x = x_raw[mask]
        y = y_raw[mask]
        l_min, l_max = min(x.min(), y.min()), max(x.max(), y.max())
        if l_min < col_lims[j]['min']: col_lims[j]['min'] = l_min
        if l_max > col_lims[j]['max']: col_lims[j]['max'] = l_max
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap=strict_purple_cmap, mincnt=1, linewidths=0.2)
        if i == 0 and j == 0: hb_list.append(hb)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        rmse = np.sqrt(np.mean((x - y)**2))
        r2 = r_value**2
        x_fit = np.linspace(l_min, l_max, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='#d62728', lw=LW_LINE, linestyle='-', label='Fit')
        stats_text = (f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nSlope = {slope:.2f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=FS_SMALL, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.3', linewidth=LW_AX))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for j in range(4):
    g_min, g_max = col_lims[j]['min'], col_lims[j]['max']
    margin = (g_max - g_min) * 0.05
    lim_min, lim_max = g_min - margin, g_max + margin
    for i in range(2):
        ax = axes[i, j]
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=LW_REF, alpha=0.6, zorder=0)
        if i == 0:
            ax.set_title(col_titles[j], fontsize=FS, fontweight='bold', pad=4)
        if j == 0:
            ax.set_ylabel(f"Emulator {unit_mon}")
            ax.annotate(selected_models[i], xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-3, 0),
                        textcoords='offset points', fontsize=FS, fontweight='bold', rotation=90, va='center', ha='right')
        else:
            ax.set_ylabel(f"Emulator {unit_yr if j == 1 else unit_mon}")
        if i == 1:
            current_unit = unit_yr if j == 1 else unit_mon
            ax.set_xlabel(f"Model {current_unit}")

fig.align_ylabels(axes[:, 0])
cbar = fig.colorbar(hb, ax=axes, orientation='horizontal', fraction=0.029, pad=0.03, aspect=30)
cbar.set_label('Data density (log scale)', fontsize=FS)
cbar.ax.tick_params(labelsize=FS_TICK, width=LW_AX)
cbar.outline.set_linewidth(LW_AX)
save_fig(fig, 'Fig4', fit_panels=True)
plt.show()
