# Fig 2: global monthly NBP (2010-2023) and annual NBP of the test years, model vs emulator.
apply_style()

COLOR_ORG = '#1a1a1a'  # model
COLOR_EMU = '#d62728'  # emulator
COLOR_SPLIT = '#7f7f7f'  # line at the start of the test period

selected_models = ['ORCHIDEE', 'CLM6.0']

fig = plt.figure(figsize=(FIG_W2, FIG_W2 * 8 / 14))
gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1], wspace=0.18, hspace=0.4)

for i, dgvm_name in enumerate(selected_models):
    cnn_data = cnn_results[dgvm_name]
    model_data = model_results[dgvm_name]
    cnn_mon = cal_mon(cnn_data)
    model_mon = cal_mon(model_data)

    t_model = pd.to_datetime(model_mon.time.values)
    mask_time = (t_model >= '2010-01-01') & (t_model < '2024-01-01')
    t_plot = t_model[mask_time]
    y_orig = model_mon.values[mask_time]
    y_emu = cnn_mon.values[mask_time]

    ax_ts = fig.add_subplot(gs[i, 0])
    ax_bar = fig.add_subplot(gs[i, 1])

    ax_ts.plot(t_plot, y_orig, color=COLOR_ORG, linewidth=LW_LINE, label='Model', zorder=2)
    ax_ts.plot(t_plot, y_emu, color=COLOR_EMU, linewidth=LW_LINE, linestyle='-', alpha=0.9, label='Emulator', zorder=3)
    ax_ts.axhline(0, color='black', linewidth=LW_AX, alpha=1.0, zorder=1)
    split_date = datetime.datetime(2022, 1, 1)
    ax_ts.axvline(split_date, color=COLOR_SPLIT, linestyle='--', linewidth=LW_REF, zorder=1)

    ylim = ax_ts.get_ylim()
    ax_ts.set_ylim(ylim[0], ylim[1]*1.15)
    ax_ts.text(datetime.datetime(2016, 1, 1), ax_ts.get_ylim()[1]*0.9, 'Training',
               color=COLOR_SPLIT, ha='center', fontsize=FS, fontweight='bold')
    ax_ts.text(datetime.datetime(2023, 1, 1), ax_ts.get_ylim()[1]*0.9, 'Test',
               color=COLOR_SPLIT, ha='center', fontsize=FS, fontweight='bold')

    mask_test = t_plot >= split_date
    y_orig_test = y_orig[mask_test]
    y_emu_test = y_emu[mask_test]
    r2 = pearson_r2(y_orig_test, y_emu_test)
    rmse = np.sqrt(np.mean((y_orig_test - y_emu_test)**2))
    stats_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}"
    ax_ts.text(0.99, 0.03, stats_text, transform=ax_ts.transAxes, ha='right', va='bottom',
               fontsize=FS_TICK, color='#333333')   # no white box: it clipped the Nov 2023 minimum of the curve

    ax_ts.set_ylabel("Global NBP (GtC month$^{-1}$)", fontweight='normal')
    ax_ts.spines['top'].set_visible(False)
    ax_ts.spines['right'].set_visible(False)
    panel_label(ax_ts, chr(97 + i*2), x=-0.08, y=1.05)

    ax_ts.set_xlim(datetime.datetime(2010, 1, 1), datetime.datetime(2024, 1, 1))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if i == 0:
        ax_ts.legend(loc='upper left', ncol=1, fontsize=FS, frameon=False)
    if dgvm_name == "ORCHIDEE":
        ax_ts.set_ylim(-1.5, 2.5)
    elif dgvm_name == "CLM6.0":
        ax_ts.set_ylim(-3.5, 5.5)
    if i == 0:
        ax_ts.yaxis.set_major_locator(MultipleLocator(1))

    df_compare = pd.DataFrame({'Org': y_orig, 'Emu': y_emu}, index=t_plot)
    df_2223 = df_compare[df_compare.index.year.isin([2022, 2023])]
    df_annual_sum = df_2223.resample('YS').sum()
    years = df_annual_sum.index.year.tolist()
    org_vals = df_annual_sum['Org'].values
    emu_vals = df_annual_sum['Emu'].values
    x_pos = np.arange(len(years))
    width = 0.35
    rects1 = ax_bar.bar(x_pos - width/2, org_vals, width, label='Model', color=COLOR_ORG, alpha=0.9)
    rects2 = ax_bar.bar(x_pos + width/2, emu_vals, width, label='Emulator', color=COLOR_EMU, alpha=0.9)
    for t in ax_bar.bar_label(rects1, padding=1.5, fmt='%.2f', fontsize=FS_SMALL) + ax_bar.bar_label(rects2, padding=1.5, fmt='%.2f', fontsize=FS_SMALL):
        t.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3))   # keeps the grid line from running through the value
    ax_bar.set_title("Annual NBP", fontsize=FS, fontweight='normal', pad=5)
    ax_bar.set_ylabel("GtC yr$^{-1}$")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(years)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.yaxis.set_ticks_position('left')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.3, linewidth=LW_AX)
    ax_bar.set_axisbelow(True)
    max_h = max(org_vals.max(), emu_vals.max())
    min_h = min(org_vals.min(), emu_vals.min())
    if max_h > 0:
        ax_bar.set_ylim(min(0, min_h*1.2), max_h * 1.25)
    else:
        ax_bar.set_ylim(min_h * 1.25, max(0, max_h*1.2))
    ax_bar.axhline(0, color='black', linewidth=LW_AX)
    panel_label(ax_bar, chr(97 + i*2 + 1), x=-0.15, y=1.05)
    if i == 0:
        ax_bar.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        ax_bar.legend(loc='upper right', ncol=1, fontsize=FS, frameon=True, facecolor='white', edgecolor='none',
                      framealpha=1, borderpad=0.2, bbox_to_anchor=(1.0, 1.0))

    ax_ts.set_title(dgvm_name, fontsize=FS, fontweight='bold', pad=6, loc='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
save_fig(fig, 'Fig2')
plt.show()
