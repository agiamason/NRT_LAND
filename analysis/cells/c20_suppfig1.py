# Supp Fig 1: as Fig 2 for the other models.
apply_style()
COLOR_ORG = '#1a1a1a'
COLOR_EMU = '#d62728'
COLOR_SPLIT = '#7f7f7f'

all_models_list = [
    'CABLE-POP', 'CARDAMOM', 'CLASSIC', 'CLM6.0', 'E3SM', 'EDv3',
    'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES',
    'LPJwsl', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM',
    'VISIT', 'iMAPLE', 'visit-ut'
]
fig1_models = ['ORCHIDEE', 'CLM6.0']
si_models = [m for m in all_models_list if m not in fig1_models]

fig = plt.figure(figsize=(FIG_W2, FIG_W2 * 17 / 20))
outer_gs = gridspec.GridSpec(6, 3, figure=fig, wspace=0.22, hspace=0.42)

for i, dgvm_name in enumerate(si_models):
    cell = outer_gs[i]
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[2.3, 1], wspace=0.45)

    cnn_mon = cal_mon(cnn_results[dgvm_name])
    model_mon = cal_mon(model_results[dgvm_name])
    t_model = pd.to_datetime(model_mon.time.values)
    mask_time = (t_model >= '2010-01-01') & (t_model < '2024-01-01')
    t_plot = t_model[mask_time]
    y_orig = model_mon.values[mask_time]
    y_emu = cnn_mon.values[mask_time]

    ax_ts = plt.subplot(inner_gs[0])
    ax_ts.plot(t_plot, y_orig, color=COLOR_ORG, linewidth=0.6, zorder=2)
    ax_ts.plot(t_plot, y_emu, color=COLOR_EMU, linewidth=0.6, alpha=0.9, zorder=3)
    ax_ts.axhline(0, color='black', linewidth=LW_AX, zorder=1)
    split_date = datetime.datetime(2022, 1, 1)
    ax_ts.axvline(split_date, color=COLOR_SPLIT, linestyle='--', linewidth=LW_AX, zorder=1)
    ax_ts.set_title(disp(dgvm_name), fontsize=FS, fontweight='bold', loc='left', pad=3)

    mask_test = t_plot >= split_date
    y_orig_test = y_orig[mask_test]
    y_emu_test = y_emu[mask_test]
    r2 = pearson_r2(y_orig_test, y_emu_test)
    rmse = np.sqrt(np.mean((y_orig_test - y_emu_test)**2))
    stats_line = f"$R^2$={r2:.3f}  RMSE={rmse:.2f}"

    ax_ts.set_xlim(datetime.datetime(2010, 1, 1), datetime.datetime(2024, 1, 1))
    col_group = i % 3
    if col_group == 0:
        ax_ts.set_ylabel("Global NBP\n(GtC month$^{-1}$)", fontsize=FS_SMALL, labelpad=2)
    ax_ts.spines['top'].set_visible(False)
    ax_ts.spines['right'].set_visible(False)
    is_bottom_row = (i >= 14)
    if is_bottom_row:
        ax_ts.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax_ts.set_xticklabels([])

    ax_bar = plt.subplot(inner_gs[1])
    df_compare = pd.DataFrame({'Org': y_orig, 'Emu': y_emu}, index=t_plot)
    df_2223 = df_compare[df_compare.index.year.isin([2022, 2023])]
    df_annual_sum = df_2223.resample('YS').sum()
    years = df_annual_sum.index.year.tolist()
    org_vals = df_annual_sum['Org'].values
    emu_vals = df_annual_sum['Emu'].values
    x_pos = np.arange(len(years))
    width = 0.4
    r_org = ax_bar.bar(x_pos - width/2, org_vals, width, color=COLOR_ORG, alpha=0.9)
    r_emu = ax_bar.bar(x_pos + width/2, emu_vals, width, color=COLOR_EMU, alpha=0.9)
    # values on the bars as in Fig 2 (a bar near zero is otherwise read as missing); vertical text so that the four labels fit
    ax_bar.bar_label(r_org, padding=1, fmt='%.2f', fontsize=FS_SMALL - 1, rotation=90)
    ax_bar.bar_label(r_emu, padding=1, fmt='%.2f', fontsize=FS_SMALL - 1, rotation=90)
    _lo = min(0.0, org_vals.min(), emu_vals.min()); _hi = max(0.0, org_vals.max(), emu_vals.max()); _rg = _hi - _lo
    ax_bar.set_ylim(_lo - (0.5 * _rg if _lo < 0 else 0.0), _hi + (0.5 * _rg if _hi > 0 else 0.05 * _rg))
    ax_bar.set_ylabel("GtC yr$^{-1}$", fontsize=FS_SMALL, labelpad=2)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(['2022', '2023'])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3, linewidth=LW_AX)
    ax_bar.set_axisbelow(True)
    ax_bar.axhline(0, color='black', linewidth=LW_AX)
    # header row of the model block: same baseline as the model name (title pad = 3 pt)
    ax_bar.annotate(stats_line, xy=(1.0, 1.0), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points',
                    ha='right', va='baseline', fontsize=FS_SMALL)

line_org = plt.Line2D([], [], color=COLOR_ORG, linewidth=LW_LINE, label='Model (monthly)')
line_emu = plt.Line2D([], [], color=COLOR_EMU, linewidth=LW_LINE, label='Emulator (monthly)')
bar_org = mpatches.Patch(color=COLOR_ORG, alpha=0.9, label='Model (annual)')
bar_emu = mpatches.Patch(color=COLOR_EMU, alpha=0.9, label='Emulator (annual)')
fig.legend(handles=[line_org, line_emu, bar_org, bar_emu], loc='lower center',
           bbox_to_anchor=(0.5, 0.001), ncol=4, fontsize=FS, frameon=False)
plt.subplots_adjust(top=0.96, bottom=0.07, left=0.07, right=0.98)
save_fig(fig, 'SuppFig1')
plt.show()
