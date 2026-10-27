# Shared definitions used by every figure, number and table: periods; unit conversion with the actual number of days
# in each month (sec_per_month, to_gC_month, annual_gC); global totals (cal_mon); the valid-cell mask; each dataset's own
# 2010-2021 monthly climatology; R2 helpers (R2 = squared Pearson correlation; R2_SSE = 1 - SSE/SST); TRENDY display
# names; and one figure style at final print size. save_fig resizes the canvas so that the exported file is exactly
# 183 mm (double column) or 89 mm (single column) wide and at most 170 mm high; with fit_panels=True the height is also
# set so that square scatter / Q-Q panels fill the frame. mathtext uses Arial (mathtext.fontset = 'custom').
import matplotlib as mpl
import struct

# ---- periods ----
REF_SLICE = slice('2010-01-01', '2021-12-31')    # climatology base period for anomalies
TEST_SLICE = slice('2022-01-01', '2023-12-31')   # recursive test period
FULL_SLICE = slice('2010-01-01', '2023-12-31')
# OUT_DIR (all outputs) and the input paths are set in analysis/config.py

# ---- units: actual number of days in each month ----
def sec_per_month(da):
    t = pd.DatetimeIndex(pd.to_datetime(da['time'].values))
    return xr.DataArray(t.days_in_month.values*86400.0, coords={'time': da['time']}, dims='time')

def to_gC_month(da):
    return da*sec_per_month(da)*1000.0

def annual_gC(da):
    return to_gC_month(da).groupby('time.year').sum('time', min_count=12)

def cal_mon(da):
    area = seaflux.area.area_grid(da.lat, da.lon, return_dataarray=True)
    return (da*area*sec_per_month(da)/1e12).sum(dim=['lat','lon'])

# ---- valid mask: complete over 2010-2023 and temporal sd > 1e-10 kg m-2 s-1 ----
def valid_mask(da):
    d = da.sel(time=FULL_SLICE)
    return d.notnull().all('time') & (d.std('time') > 1e-10)

def calculate_anomaly(x, monthly_mean):
    month = x['time.month']
    return x - monthly_mean.sel(month=month)

# ---- each dataset minus its own 2010-2021 monthly climatology ----
def clim_ref(da):
    return da.sel(time=REF_SLICE).groupby('time.month').mean('time')

def seasonal_cycle_test(da):
    return da.sel(time=TEST_SLICE).groupby('time.month').mean('time')

# ---- statistics: R2 = squared Pearson correlation; R2_SSE = 1 - SSE/SST ----
def pearson_r2(x, y):
    x = np.asarray(x, dtype=float).ravel(); y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)

def r2_sse(obs, pred):
    obs = np.asarray(obs, dtype=float).ravel(); pred = np.asarray(pred, dtype=float).ravel()
    ok = np.isfinite(obs) & np.isfinite(pred); obs, pred = obs[ok], pred[ok]
    return float(1.0 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))

# ---- TRENDY display names ----
DISPLAY_NAME = {'visit-ut': 'VISIT-UT'}
def disp(name):
    return DISPLAY_NAME.get(name, name)

# ---- shared figure style at final print size ----
MM = 1 / 25.4
FIG_W2, FIG_W1 = 183 * MM, 89 * MM                        # double and single column width (inch)
FS, FS_TICK, FS_SMALL, FS_PANEL = 7, 6.5, 6, 8           # text, tick labels, small text, panel labels (pt)
LW_AX, LW_LINE, LW_REF = 0.5, 0.8, 0.6                   # axes, data lines, reference lines (pt)

import matplotlib._mathtext as _mathtext_mod
def _xheight_from_glyph(self, fontname, fontsize, dpi):
    # Arial has a PCLT table whose xHeight matplotlib 3.8 divides by 64 as if it were 26.6 pixels; that gives about
    # twice the real x-height, so superscripts sit too high with a gap. Use the height of the 'x' glyph instead.
    return self.get_metrics(fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi).iceberg
_mathtext_mod.TruetypeFonts.get_xheight = _xheight_from_glyph

def apply_style(mathtext='it'):
    # mathtext: 'it' = matplotlib default; 'regular' = Figs 5, 6 and Supp Figs 7-9
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': FS,
        'axes.titlesize': FS, 'axes.labelsize': FS, 'legend.fontsize': FS,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'axes.linewidth': LW_AX, 'lines.linewidth': LW_LINE, 'patch.linewidth': LW_AX,
        'xtick.major.width': LW_AX, 'ytick.major.width': LW_AX, 'xtick.minor.width': LW_AX, 'ytick.minor.width': LW_AX,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2, 'axes.titlepad': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.default': mathtext,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.dpi': 100, 'savefig.dpi': 600,
    })

def panel_label(ax, letter, x=-0.10, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='right')

def _export_size(fig, pad):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width + 2 * pad, bb.height + 2 * pad

def _square_panels(fig):
    # equal-aspect panels (scatter / Q-Q); colour bars and blank slots have aspect 'auto'
    return [ax for ax in fig.axes if ax.get_visible() and ax.axison and ax.get_aspect() != 'auto']

def _spare(fig):
    # mean unused width and height (inch) of the layout box around each square panel, and the panel grid shape
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    axs = _square_panels(fig)
    if not axs:
        return 0.0, 0.0, 1, 1
    o = [ax.get_position(original=True) for ax in axs]
    p = [ax.get_position(original=False) for ax in axs]
    sw = float(np.mean([(a.width - b.width) * W for a, b in zip(o, p)]))
    sh = float(np.mean([(a.height - b.height) * H for a, b in zip(o, p)]))
    return sw, sh, len({round(a.y0, 3) for a in o}), len({round(a.x0, 3) for a in o})

def save_fig(fig, name, width_mm=183, max_h_mm=170, fit_panels=False, **kw):
    # Font sizes are fixed in pt, so the canvas (not the text) is resized until the cropped export is
    # width_mm wide and at most max_h_mm high. fit_panels=True (square scatter / Q-Q panels): the height is
    # also set so the square panels fill the frame instead of leaving white gaps; if that would exceed
    # max_h_mm, the height is capped and the width is reduced to the panels instead (no gaps either way).
    kw.setdefault('bbox_inches', 'tight'); kw.setdefault('pad_inches', 0.02)
    pad = kw['pad_inches']
    target_w = None if width_mm is None else width_mm * MM
    for _ in range(12):                                   # phase 1: target width, panels filling the frame
        W, H = fig.get_size_inches()
        w, h = _export_size(fig, pad)
        dW = 0.0 if target_w is None else target_w - w
        dH = 0.0
        if fit_panels:
            sw, sh, nr, nc = _spare(fig)
            dH = nr * (sw - sh)
        if abs(dW) < 0.1 * MM and abs(dH) < 0.2 * MM:
            break
        fig.set_size_inches(max(W + dW, 1.0), max(H + dH, 1.0), forward=False)
    w, h = _export_size(fig, pad)
    if h > max_h_mm * MM:                                 # phase 2: height cap
        for _ in range(12):
            W, H = fig.get_size_inches()
            w, h = _export_size(fig, pad)
            dH = min(0.0, max_h_mm * MM - h)
            dW = 0.0
            if fit_panels:
                target_w = None                           # panels now set by the height: drop the spare width
                sw, sh, nr, nc = _spare(fig)
                dW = -nc * (sw - sh)
            elif target_w is not None:
                dW = target_w - w
            if abs(dH) < 0.1 * MM and abs(dW) < 0.2 * MM:
                break
            fig.set_size_inches(max(W + dW, 1.0), max(H + dH, 1.0), forward=False)
    # text extents differ slightly between screen and 600 dpi: correct on the saved PNG itself
    for _ in range(4):
        fig.savefig(OUT_DIR + name + '.png', dpi=600, **kw)
        with open(OUT_DIR + name + '.png', 'rb') as fh:
            w_px, h_px = struct.unpack('>II', fh.read(24)[16:24])
        w_mm, h_mm = w_px / 600 * 25.4, h_px / 600 * 25.4
        dw = 0.0 if target_w is None else target_w / MM - w_mm
        dh = min(0.0, max_h_mm - h_mm)
        if abs(dw) < 0.3 and dh > -0.3:
            break
        W, H = fig.get_size_inches()
        fig.set_size_inches(W + dw * MM, H + dh * MM, forward=False)
    fig.savefig(OUT_DIR + name + '.pdf', **kw)
    print(f'Saved {OUT_DIR}{name}.png and .pdf ({w_mm:.1f} mm x {h_mm:.1f} mm)')

apply_style()
