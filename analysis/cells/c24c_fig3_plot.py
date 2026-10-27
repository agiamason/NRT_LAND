# Fig 3, part 3: heat maps of R2 (a) and absolute RMSE (b, each RMSE column on its own colour scale); also writes
# Fig3_R2_two_scales.txt.
with open(OUT_DIR + 'Fig3_R2_two_scales.txt', 'w') as fh:
    fh.write("Fig 3 columns: R2 at grid scale and at global scale (valid_mask_final; 2022-2023 test period; R2 = squared Pearson correlation)\n")
    fh.write("grid_median = per-pixel R2 over time, median over valid pixels (reference temporal sd > 1e-10);\n")
    fh.write("grid_pooled = R2 of all valid pixel-time samples pooled together; MeanAnnual_grid_spatial = R2 of the two 2022-2023 mean annual NBP maps (only one sample per pixel);\n")
    fh.write("global = R2 of the global-total time series (cal_mon, GtC month-1): monthly n=24, seasonal cycle n=12 (2022-2023 mean), monthly anomalies n=24, quarterly anomalies n=8; mean annual NBP has no global-scale time series (n=2), left blank.\n")
    fh.write("The heatmap shows the MEAN over pixels of the per-pixel R2 for the temporal columns.\n\n")
    fh.write(scale_df.to_string(float_format=lambda v: f"{v:.3f}"))
    fh.write("\n\nMedian over the 19 models (Ensemble-median row excluded):\n")
    fh.write(scale_df.drop(index='Ensemble median').median().to_string(float_format=lambda v: f"{v:.3f}"))
print(scale_df.round(3).to_string())

# ================= 2. Heatmaps =================
# lighter ends so no cell is near-black; Blues up to 80 % meets RdBu_r's blue at 10 %, so both panels share one blue
from matplotlib.colors import LinearSegmentedColormap
CMAP_R2 = LinearSegmentedColormap.from_list('Blues_0_80', plt.get_cmap('Blues')(np.linspace(0.0, 0.8, 256)))
CMAP_REL = LinearSegmentedColormap.from_list('RdBu_r_10_90', plt.get_cmap('RdBu_r')(np.linspace(0.1, 0.9, 256)))

def _bw_text(ax):
    # seaborn writes white on dark cells and dark grey (.15) on light cells; make the dark grey pure black
    for t in ax.texts:
        if t.get_color() not in ('w', 'white', '#ffffff'):
            t.set_color('black')

def heat_r2(ax):
    # colour bar in an inset below the heatmap, so the heatmap axes keep their size
    cax = ax.inset_axes([0.2, -0.075, 0.6, 0.022])
    sns.heatmap(df_r2, ax=ax, annot=True, fmt=".2f", cmap=CMAP_R2, vmin=0, vmax=1,
                cbar_ax=cax, cbar_kws={'orientation': 'horizontal'},
                linewidths=0.5, linecolor='white', annot_kws={"size": FS_SMALL, "weight": "normal"})
    cb = ax.collections[0].colorbar
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cb.ax.tick_params(labelsize=FS_TICK, width=LW_AX)
    cb.set_label('$R^2$', fontsize=FS)
    cb.outline.set_linewidth(LW_AX)
    _bw_text(ax)

def style_heat(ax, labels):
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_xticklabels(labels, rotation=45, ha='left', rotation_mode='anchor', fontweight='bold', fontsize=FS_TICK)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontsize(FS_TICK)

# R2 (a) and absolute RMSE (b, Gleckler-style portrait diagram)
rmse_titles = ['Monthly NBP\n(gC m$^{-2}$ month$^{-1}$)', 'Mean annual NBP\n(gC m$^{-2}$ yr$^{-1}$)',
               'Mean seasonal cycle\n(gC m$^{-2}$ month$^{-1}$)', 'Monthly anomalies\n(gC m$^{-2}$ month$^{-1}$)',
               'Quarterly anomalies\n(gC m$^{-2}$ month$^{-1}$)']
models19 = [n for n in df_rmse.index if n != 'Ensemble median']
med19 = df_rmse.loc[models19].median(axis=0)              # per-column median over the 19 models only
df_rel = (df_rmse - med19) / med19                         # Ensemble-median row coloured with the same formula
print('Median RMSE of the 19 models per column:'); print(med19.round(3).to_string())
print('Relative RMSE range:', round(float(np.nanmin(df_rel.values)), 3), 'to', round(float(np.nanmax(df_rel.values)), 3))
annot_rmse = np.array([[f"{v:.0f}" if j == 1 else f"{v:.1f}" for j, v in enumerate(row)] for row in df_rmse.values])
fig_ii, axes = plt.subplots(1, 2, figsize=(FIG_W2, 125 * MM), sharey=True)
heat_r2(axes[0])
style_heat(axes[0], col_names)
cax_b = axes[1].inset_axes([0.2, -0.075, 0.6, 0.022])
sns.heatmap(df_rel, ax=axes[1], annot=annot_rmse, fmt='', cmap=CMAP_REL, vmin=-1, vmax=1,
            cbar_ax=cax_b, cbar_kws={'orientation': 'horizontal'},
            linewidths=0.5, linecolor='white', annot_kws={"size": FS_SMALL, "weight": "normal"})
style_heat(axes[1], rmse_titles)
_bw_text(axes[1])
axes[1].tick_params(axis='y', left=False)
cb_b = axes[1].collections[0].colorbar
cb_b.set_ticks([-1, -0.5, 0, 0.5, 1])
cb_b.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace('-', '\u2212')))
cb_b.ax.tick_params(labelsize=FS_TICK, width=LW_AX)
cb_b.set_label('RMSE relative to the multi-model median', fontsize=FS)
cb_b.outline.set_linewidth(LW_AX)
panel_label(axes[0], 'a', x=-0.02, y=1.02)
panel_label(axes[1], 'b', x=-0.02, y=1.02)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
save_fig(fig_ii, 'Fig3')
plt.show()
