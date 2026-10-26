# Fig 5: maps of mean annual NBP and of monthly anomalies (model, emulator, emulator - model) with the spatial R2 of
# each map (also written to Fig5_spatial_R2.txt).
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def prepare_map_data(model_name):
    raw_mod = model_results[model_name]
    raw_emu = cnn_results[model_name]
    r1_mod = annual_gC(raw_mod.sel(time=TEST_SLICE)).mean('year')
    r1_emu = annual_gC(raw_emu.sel(time=TEST_SLICE)).mean('year')
    da_mod, da_emu = to_gC_month(raw_mod), to_gC_month(raw_emu)
    anom_mod_full = calculate_anomaly(da_mod.sel(time=TEST_SLICE), clim_ref(da_mod))
    anom_emu_full = calculate_anomaly(da_emu.sel(time=TEST_SLICE), clim_ref(da_emu))
    mask_jfm = anom_mod_full.time.dt.month.isin([1, 2, 3])
    r2_mod = anom_mod_full.where(mask_jfm).mean('time')
    r2_emu = anom_emu_full.where(mask_jfm).mean('time')
    mask_jas = anom_mod_full.time.dt.month.isin([7, 8, 9])
    r3_mod = anom_mod_full.where(mask_jas).mean('time')
    r3_emu = anom_emu_full.where(mask_jas).mean('time')
    return [(r1_mod, r1_emu), (r2_mod, r2_emu), (r3_mod, r3_emu)]

apply_style(mathtext='regular')
plt.rcParams.update({'savefig.pad_inches': 0.02})
nbp_cmap = plt.get_cmap('BrBG')   # brown = source, blue-green = sink
unit_mon, unit_yr = "(gC m$^{-2}$ month$^{-1}$)", "(gC m$^{-2}$ yr$^{-1}$)"
abs_limits = [200, 40, 40]
diff_limits = [150, 30, 30]
row_names = ['Mean annual', 'JFM anomaly', 'JAS anomaly']
selected_models = ['ORCHIDEE', 'CLM6.0']

fig = plt.figure(figsize=(FIG_W2, FIG_W2 * 16 / 20))
gs_main = gridspec.GridSpec(2, 1, hspace=0.12, right=0.835, top=0.92, bottom=0.05)
spatial_rows = []

for m_idx, m_name in enumerate(selected_models):
    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_main[m_idx], wspace=0.01, hspace=0.05)
    ims, axes_all = {}, {}
    rows_data = prepare_map_data(m_name)
    panel_letter = 'a' if m_idx == 0 else 'b'
    for r_idx in range(3):
        mod_data, emu_data = rows_data[r_idx]
        diff_data = emu_data - mod_data
        current_data = [mod_data, emu_data, diff_data]
        xv, yv = mod_data.values.ravel(), emu_data.values.ravel()
        ok = np.isfinite(xv) & np.isfinite(yv)
        r2_sp = pearson_r2(xv[ok], yv[ok])
        spatial_rows.append({'Model': m_name, 'Row': row_names[r_idx], 'spatial_R2': r2_sp, 'n_cells': int(ok.sum())})
        for c_idx in range(3):
            ax = fig.add_subplot(gs_sub[r_idx, c_idx], projection=ccrs.Robinson())
            ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
            cmap = nbp_cmap if c_idx < 2 else 'RdBu_r'
            lim = abs_limits[r_idx] if c_idx < 2 else diff_limits[r_idx]
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
            im = current_data[c_idx].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                                          add_colorbar=False, add_labels=False, rasterized=True)
            ax.set_title("")
            if c_idx == 1: ims[f'abs_{r_idx}'] = im
            if c_idx == 2: ims[f'diff_{r_idx}'] = im
            axes_all[f'{r_idx}_{c_idx}'] = ax
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='black')
            if c_idx == 1:
                # lower-left of the map, over the open South Pacific (about 41-31 S, inside the Robinson frame)
                ax.text(0.065, 0.13, rf"$\mathit{{R}}^2$ = {r2_sp:.2f}", transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=FS_TICK, color='#333333')
            if m_idx == 0 and r_idx == 0:
                col_titles = ['Model', 'Emulator', 'Emulator − model']
                ax.set_title(col_titles[c_idx], fontsize=FS, fontweight='bold', pad=4)
            if c_idx == 0:
                ax.text(-0.08, 0.5, row_names[r_idx], transform=ax.transAxes,
                        fontsize=FS, fontweight='bold', rotation=90, va='center', ha='center')
                if r_idx == 1:
                    ax.text(-0.20, 0.5, m_name, transform=ax.transAxes,
                            fontsize=FS, fontweight='bold', rotation=90, va='center', ha='center')
                if r_idx == 0:
                    ax.text(-0.25, 1, panel_letter, transform=ax.transAxes,
                            fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='center')

    def add_cbar_row0_and_shared(im_row0, im_shared, label_row0, label_shared, x_pos):
        pos0 = axes_all['0_1'].get_position()
        cax0 = fig.add_axes([x_pos, pos0.y0, 0.012, pos0.height])
        cb0 = fig.colorbar(im_row0, cax=cax0)
        cb0.set_label(label_row0, fontsize=FS_SMALL, fontweight='bold')
        cb0.ax.tick_params(labelsize=FS_SMALL, width=LW_AX)
        cb0.outline.set_linewidth(LW_AX)
        pos1 = axes_all['1_1'].get_position()
        pos2 = axes_all['2_1'].get_position()
        y_center = 0.5 * ((pos1.y0 + pos1.height/2) + (pos2.y0 + pos2.height/2))
        y0_shared = y_center - pos1.height/2
        cax_shared = fig.add_axes([x_pos, y0_shared, 0.012, pos1.height])
        cb_shared = fig.colorbar(im_shared, cax=cax_shared)
        cb_shared.set_label(label_shared, fontsize=FS_SMALL, fontweight='bold')
        cb_shared.ax.tick_params(labelsize=FS_SMALL, width=LW_AX)
        cb_shared.outline.set_linewidth(LW_AX)

    abs_labels = [f"NBP {unit_yr}", f"NBP {unit_mon}", f"NBP {unit_mon}"]
    diff_labels = [f"Emulator − model\n{unit_yr}", f"Emulator − model\n{unit_mon}", f"Emulator − model\n{unit_mon}"]
    add_cbar_row0_and_shared(ims['abs_0'], ims['abs_1'], abs_labels[0], abs_labels[1], 0.85)
    add_cbar_row0_and_shared(ims['diff_0'], ims['diff_1'], diff_labels[0], diff_labels[1], 0.925)

save_fig(fig, 'Fig5')
spatial_df = pd.DataFrame(spatial_rows)
with open(OUT_DIR + 'Fig5_spatial_R2.txt', 'w') as fh:
    fh.write("Fig 5: spatial R2 between emulator and reference maps (squared Pearson correlation over valid_mask_final cells, unweighted), 2022-2023\n")
    fh.write("Mean annual = mean of the 2022 and 2023 annual totals (gC m-2 yr-1); JFM / JAS anomaly = mean monthly anomaly in those months (gC m-2 month-1), each dataset minus its own 2010-2021 climatology\n\n")
    fh.write(spatial_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(spatial_df.round(3).to_string(index=False))
plt.show()
