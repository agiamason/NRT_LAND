# Supp Figs 3-5: as Fig 4 for the other 17 models.
apply_style()

def get_truncated_purples():
    cmap = plt.get_cmap('Purples')
    minval = 0.5  # darker start of the colour map
    maxval = 1.0
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, 256)))
    return new_cmap

strict_purple_cmap = get_truncated_purples()

def scatter_component_pairs(dgvm_name):
    """Four components in gC m-2 (month-1 or yr-1). Day-accurate units; mean seasonal cycle from the
    2022-2023 test period (sc_model / sc_cnn); clim_model / clim_cnn stay 2010-2021 as the anomaly base."""
    raw_model, raw_cnn = model_results[dgvm_name], cnn_results[dgvm_name]
    da_model, da_cnn = to_gC_month(raw_model), to_gC_month(raw_cnn)
    # [Col 1] Monthly NBP (Test Period)
    x1 = da_model.sel(time=test_slice).values.flatten()
    y1 = da_cnn.sel(time=test_slice).values.flatten()
    # [Col 2] Mean Annual / Spatial (Test Period), mean of the calendar-year totals
    x2 = annual_gC(raw_model.sel(time=test_slice)).mean('year').values.flatten()
    y2 = annual_gC(raw_cnn.sel(time=test_slice)).mean('year').values.flatten()
    # [Col 3] Mean Seasonal Cycle (Test Period 2022-2023)
    sc_model = da_model.sel(time=test_slice).groupby('time.month').mean('time')
    sc_cnn = da_cnn.sel(time=test_slice).groupby('time.month').mean('time')
    x3 = sc_model.values.flatten()
    y3 = sc_cnn.values.flatten()
    # [Col 4] Monthly Anomalies (Test Period), anomaly base = own 2010-2021 climatology
    clim_model = da_model.sel(time=ref_slice).groupby('time.month').mean('time')
    clim_cnn = da_cnn.sel(time=ref_slice).groupby('time.month').mean('time')
    anom_model = calculate_anomaly(da_model.sel(time=test_slice), clim_model)
    anom_cnn = calculate_anomaly(da_cnn.sel(time=test_slice), clim_cnn)
    x4 = anom_model.values.flatten()
    y4 = anom_cnn.values.flatten()
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

unit_mon = "(gC m$^{-2}$ month$^{-1}$)"
unit_yr = "(gC m$^{-2}$ yr$^{-1}$)"
main_models = ['ORCHIDEE', 'CLM6.0']
all_models = sorted(list(model_results.keys()))
si_models = [m for m in all_models if m not in main_models]
batches = [si_models[0:6], si_models[6:12], si_models[12:]]
col_titles = ['Monthly NBP', 'Mean annual NBP', 'Mean seasonal cycle', 'Monthly anomalies']
ref_slice = REF_SLICE
test_slice = TEST_SLICE

def plot_si_batch(models_batch, fig_idx):
    n_rows = len(models_batch)
    page_h_mm = 25 + 34.2 * n_rows           # full supplementary page: 230 mm for 6 rows, same row height for 5 rows
    fig_h = page_h_mm * MM
    fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(FIG_W2, fig_h), constrained_layout=True)
    hb_list = []
    for i, dgvm_name in enumerate(models_batch):
        data_pairs = scatter_component_pairs(dgvm_name)
        for j, (x_raw, y_raw) in enumerate(data_pairs):
            ax = axes[i, j]
            mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            x, y = x_raw[mask], y_raw[mask]
            if len(x) > 0:
                l_min, l_max = min(x.min(), y.min()), max(x.max(), y.max())
                margin = (l_max - l_min) * 0.05
                current_min, current_max = l_min - margin, l_max + margin
            else:
                current_min, current_max = 0, 1
            hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap=strict_purple_cmap, mincnt=1, linewidths=0.2)
            if i == 0 and j == 0: hb_list.append(hb)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            rmse = np.sqrt(np.mean((x - y)**2))
            r2 = r_value**2
            ax.set_xlim(current_min, current_max)
            ax.set_ylim(current_min, current_max)
            ax.set_aspect('equal')
            ax.plot([current_min, current_max], [current_min, current_max], 'k--', lw=LW_REF, alpha=0.6, zorder=0)
            x_fit = np.linspace(current_min, current_max, 100)
            ax.plot(x_fit, slope * x_fit + intercept, color='#d62728', lw=LW_LINE, zorder=3)
            stats_text = (f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nSlope = {slope:.2f}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=FS_SMALL, va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.25', linewidth=LW_AX))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=FS, fontweight='bold', pad=4)
            if j == 0:
                ax.set_ylabel(f"Emulator {unit_mon}", fontsize=FS_SMALL, labelpad=2)
                ax.annotate(disp(models_batch[i]), xy=(0, 0.5), xycoords=ax.yaxis.label, xytext=(-3, 0),
                            textcoords='offset points', fontsize=FS, fontweight='bold', rotation=90, va='center', ha='right')
            else:
                ax.set_ylabel(f"Emulator {unit_yr if j == 1 else unit_mon}", fontsize=FS_SMALL, labelpad=2)
            if i == n_rows - 1:
                current_unit = unit_yr if j == 1 else unit_mon
                ax.set_xlabel(f"Model {current_unit}", fontsize=FS_SMALL, labelpad=2)
    fig.align_ylabels(axes[:, 0])
    cbar = fig.colorbar(hb_list[0], ax=axes, orientation='horizontal', fraction=0.03, pad=0.02, aspect=28, shrink=0.45)
    cbar.set_label('Data density (log scale)', fontsize=FS)
    cbar.ax.tick_params(labelsize=FS_TICK, width=LW_AX)
    cbar.outline.set_linewidth(LW_AX)
    save_fig(fig, f'SuppFig{fig_idx}', max_h_mm=page_h_mm, fit_panels=True)
    plt.close()

for idx, batch in enumerate(batches):
    plot_si_batch(batch, idx + 3)   # Supp Figs 3, 4, 5
