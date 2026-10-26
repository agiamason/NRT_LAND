# Supp Fig 2: cross-model scatter (one point per model) for the ten regions; a-j mean annual NBP, k-t amplitude of the
# mean seasonal cycle, 2022-2023. Data: SuppFig2_values.csv (steps/regional_fit_for_purpose.py). The R2/slope box sits
# in the upper-left corner of every panel; where it would cover a point, the common x/y range of that panel is extended
# upwards until all points are clear (x and y keep the same range, so the 1:1 line stays the diagonal).
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import linregress
apply_style()
v10 = pd.read_csv(OUT_DIR + 'SuppFig2_values.csv')
R10 = ['North America', 'Europe', 'West Asia', 'Russia', 'East Asia', 'South Asia', 'Southeast Asia', 'Latin America', 'Oceania', 'Africa']
Q10 = [('mean_annual_2022_2023', 'Mean annual NBP, 2022–2023', 'GtC yr$^{-1}$'),
       ('seasonal_amplitude_2022_2023', 'Amplitude of the mean seasonal cycle, 2022–2023', 'GtC month$^{-1}$')]
fig = plt.figure(figsize=(FIG_W2, FIG_W2 * 0.86), layout='constrained')
sfs = fig.subfigures(2, 1, hspace=0.02)
panels = []
for k, (q, title, unit) in enumerate(Q10):
    sf = sfs[k]; sf.suptitle(title, fontsize=FS, fontweight='bold')
    axs = sf.subplots(2, 5)
    for i, reg in enumerate(R10):
        ax = axs[i // 5, i % 5]
        d = v10[(v10.Region == reg) & (v10.quantity == q)]
        x, y = d['model'].values, d['emulator'].values
        res = linregress(x, y)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.08 * (hi - lo)
        ax.axline((lo, lo), slope=1, color='k', linestyle='--', lw=LW_REF, alpha=0.6, zorder=0)
        ax.scatter(x, y, s=9, color='#1f77b4', edgecolors='none', alpha=0.85, zorder=2)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(reg, fontsize=FS, fontweight='bold', pad=3)
        lg = ax.legend([Line2D([], [], linestyle='none')], [f"$R^2$ = {res.rvalue ** 2:.{3 if k == 1 else 2}f}\nSlope = {res.slope:.2f}"], loc='upper left',
                       handlelength=0, handletextpad=0, borderpad=0.35, borderaxespad=0.3, fontsize=FS_SMALL,
                       frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')
        lg.get_frame().set_linewidth(LW_AX)
        panel_label(ax, chr(97 + 10 * k + i), x=-0.10, y=1.03)
        if i % 5 == 0: ax.set_ylabel(f'Emulator ({unit})', fontsize=FS_SMALL)
        if i // 5 == 1: ax.set_xlabel(f'Model ({unit})', fontsize=FS_SMALL)
        panels.append((chr(97 + 10 * k + i), reg, ax, lg, x, y, lo - pad, hi + pad))
    sf.align_ylabels(axs[:, 0])

def _hits(ax, lg, x, y):
    # points (marker radius + 1 px) whose centre falls inside the text box, in display coordinates of the current figure size
    bb = lg.get_window_extent()
    pts = ax.transData.transform(np.column_stack([x, y]))
    r = np.sqrt(9.0) / 2 * fig.dpi / 72 + 1
    return int(((pts[:, 0] > bb.x0 - r) & (pts[:, 0] < bb.x1 + r) & (pts[:, 1] > bb.y0 - r) & (pts[:, 1] < bb.y1 + r)).sum())

save_fig(fig, 'SuppFig2', fit_panels=True)
extended = {}
for rnd in range(3):                      # the export size can change slightly after a limit change, so check again after saving
    changed = False
    fig.canvas.draw()
    for lab, reg, ax, lg, x, y, l0, h0 in panels:
        n = 0
        while _hits(ax, lg, x, y) > 0 and n < 40:
            a, b = ax.get_xlim(); b = b + 0.05 * (b - a)
            ax.set_xlim(a, b); ax.set_ylim(a, b); fig.canvas.draw(); n += 1; changed = True
        if n:
            a, b = ax.get_xlim(); extended[lab] = (reg, round(l0, 3), round(h0, 3), round(b, 3))
    if not changed:
        break
    save_fig(fig, 'SuppFig2', fit_panels=True)
fig.canvas.draw()
print('panels whose range was extended (label, region, lower limit, old upper limit, new upper limit):', extended)
print('points under a text box after the final export:', {lab: _hits(ax, lg, x, y) for lab, reg, ax, lg, x, y, l0, h0 in panels})
print('Supp Fig 2 written')
