"""Supp Fig 10: precipitation, fire emissions and vegetation carbon in the 36 excluded cells of western Venezuela,
2010-2024 (36-cell means; a annual precipitation, b fire emissions, c vegetation carbon), in the style of the other
figures (cells/c03_definitions.py), exported 183 mm wide.

    python plot_supp_fig10.py crujra_precip.json extract_inputs.json OUT   (writes OUT.png, OUT.pdf and OUT_values.csv)
"""
import json, struct, sys, numpy as np, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._mathtext as _mt
_mt.TruetypeFonts.get_xheight = lambda self, f, s, d: self.get_metrics(f, mpl.rcParams['mathtext.default'], 'x', s, d).iceberg
L = json.load(open(sys.argv[1])); E = json.load(open(sys.argv[2]))
OUT = sys.argv[3]                                  # output path without extension
MM = 1 / 25.4
FS, FS_TICK, FS_SMALL, FS_PANEL = 7, 6.5, 6, 8
LW_AX, LW_LINE = 0.5, 0.8
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'legend.fontsize': FS_SMALL,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'axes.linewidth': LW_AX, 'lines.linewidth': LW_LINE, 'patch.linewidth': LW_AX,
    'xtick.major.width': LW_AX, 'ytick.major.width': LW_AX, 'xtick.minor.width': LW_AX, 'ytick.minor.width': LW_AX,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2, 'axes.titlepad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.default': 'regular',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 100, 'savefig.dpi': 600})

def panel_label(ax, letter, x=-0.10, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='right')

fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), constrained_layout=True)
yrs = list(range(2010, 2024))
def mean_cells(d):
    return {int(y): float(np.nanmean([x for x in v if x is not None])) for y, v in d.items()}
MK = dict(marker='o', ms=2)

ax = axes[0]
p23, p24 = mean_cells(L['v2.3']['cells']), mean_cells(L['v2.4']['cells'])
ct = {int(y): v['pre_mm'] for y, v in E['crut408'].items()}
er = {int(y): v for y, v in E['era5'].items() if int(y) <= 2023}
for d, lab, c, ls in [(er, 'ERA5 (emulator input)', '#1a1a1a', '-'), (ct, 'CRU TS 4.08 (basis of v2.5, TRENDY v13)', '#2166ac', '-'),
                      (p24, 'CRU-JRA v2.4 (TRENDY v12)', '#d6604d', '-'), (p23, 'CRU-JRA v2.3 (TRENDY v11)', '#f4a582', '--')]:
    x = [y for y in yrs if y in d]
    ax.plot(x, [d[y] for y in x], color=c, ls=ls, label=lab, **MK)
ax.set_ylabel('Annual precipitation (mm yr$^{-1}$)')
ax.set_ylim(0, 5600)
ax.legend(loc='upper right', handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
panel_label(ax, 'a')

ax = axes[1]
for k, lab, c, ls in [('v11 LPX-Bern fFire', 'LPX-Bern v11', '#fdae61', '--'), ('v12 LPX-Bern fFire', 'LPX-Bern v12', '#f46d43', '-.'),
                      ('v13 LPX-Bern fFire', 'LPX-Bern v13', '#a50026', '-'), ('v11 LPJ fFire', 'LPJwsl v11', '#abd9e9', '--'),
                      ('v12 LPJwsl fFire', 'LPJwsl v12', '#74add1', '-.'), ('v13 LPJwsl fFire', 'LPJwsl v13', '#313695', '-')]:
    s = {int(y): v for y, v in E[k]['series'].items()}
    x = [y for y in yrs if y in s]
    ax.plot(x, [s[y] for y in x], color=c, ls=ls, label=lab, **MK)
ax.set_yscale('log')
ax.set_ylim(10, 3e4)
ax.set_ylabel('Fire emissions, fFire (gC m$^{-2}$ yr$^{-1}$)')
ax.legend(loc='upper left', ncol=2, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3, columnspacing=1.0)
panel_label(ax, 'b')

ax = axes[2]
for k, lab, c, ls in [('v13 LPX-Bern cVeg', 'LPX-Bern v13', '#a50026', '-'), ('v14 LPX-Bern cVeg', 'LPX-Bern v14', '#f46d43', '--'),
                      ('v13 LPJwsl cVeg', 'LPJwsl v13', '#313695', '-'), ('v14 LPJ-EOSIM cVeg', 'LPJ-EOSIM v14', '#74add1', '--')]:
    s = {int(y): v for y, v in E[k]['series'].items()}
    x = [y for y in range(2010, 2025) if y in s]
    ax.plot(x, [s[y] for y in x], color=c, ls=ls, label=lab, **MK)
ax.set_ylabel('Vegetation carbon, cVeg (kgC m$^{-2}$)')
ax.set_ylim(5, 13.5)
ax.legend(loc='lower left', handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
panel_label(ax, 'c')
for a in axes:
    a.axvspan(2020.5, 2021.5, color='0.9', zorder=0, lw=0)
    a.set_xlim(2009.4, 2024.6)
    a.set_xticks([2010, 2013, 2016, 2019, 2022])
    a.set_xlabel('Year')
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)

# exported file exactly 183 mm wide (as save_fig in cells/c03_definitions.py)
kw = dict(bbox_inches='tight', pad_inches=0.02)
target = 183.0
for _ in range(6):
    fig.savefig(OUT + '.png', dpi=600, **kw)
    with open(OUT + '.png', 'rb') as fh:
        w_px, h_px = struct.unpack('>II', fh.read(24)[16:24])
    w_mm, h_mm = w_px / 600 * 25.4, h_px / 600 * 25.4
    if abs(target - w_mm) < 0.3:
        break
    W, H = fig.get_size_inches()
    fig.set_size_inches(W + (target - w_mm) * MM, H, forward=False)
fig.savefig(OUT + '.pdf', **kw)
print(f'saved {OUT}.png and .pdf ({w_mm:.1f} mm x {h_mm:.1f} mm, {w_px} x {h_px} px)')

# the plotted values (36-cell means), as CSV
import csv
with open(OUT + '_values.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Panel', 'Series', 'Year', 'Value', 'Units'])
    for d, lab in ((er, 'ERA5'), (ct, 'CRU TS 4.08'), (p24, 'CRU-JRA v2.4'), (p23, 'CRU-JRA v2.3')):
        for y in yrs:
            if y in d:
                w.writerow(['a', lab, y, f'{d[y]:.2f}', 'mm yr-1'])
    for k, lab in (('v11 LPX-Bern fFire', 'LPX-Bern v11'), ('v12 LPX-Bern fFire', 'LPX-Bern v12'), ('v13 LPX-Bern fFire', 'LPX-Bern v13'),
                   ('v11 LPJ fFire', 'LPJwsl v11'), ('v12 LPJwsl fFire', 'LPJwsl v12'), ('v13 LPJwsl fFire', 'LPJwsl v13')):
        s = {int(y): v for y, v in E[k]['series'].items()}
        for y in yrs:
            if y in s:
                w.writerow(['b', lab, y, f'{s[y]:.4f}', 'gC m-2 yr-1'])
    for k, lab in (('v13 LPX-Bern cVeg', 'LPX-Bern v13'), ('v14 LPX-Bern cVeg', 'LPX-Bern v14'),
                   ('v13 LPJwsl cVeg', 'LPJwsl v13'), ('v14 LPJ-EOSIM cVeg', 'LPJ-EOSIM v14')):
        s = {int(y): v for y, v in E[k]['series'].items()}
        for y in range(2010, 2025):
            if y in s:
                w.writerow(['c', lab, y, f'{s[y]:.4f}', 'kgC m-2'])
print(f'saved {OUT}_values.csv')
