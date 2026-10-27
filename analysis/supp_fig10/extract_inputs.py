"""Supp Fig 10, other inputs: 36-cell means of ERA5 annual precipitation (the emulator input, cells aligned by grid index as in
the emulator), CRU TS 4.08 annual precipitation (and station counts), and TRENDY S2 fire emissions (fFire, gC m-2 yr-1)
and vegetation carbon (cVeg, kgC m-2) of LPX-Bern and LPJwsl / LPJ-EOSIM in TRENDY v11-v14.

    python extract_inputs.py supp_fig10_inputs.json extract_inputs.json
"""
import warnings; warnings.filterwarnings("ignore")
import gzip, json, os, sys, numpy as np, xarray as xr, netCDF4, cftime
CFG = json.load(open(sys.argv[1]))   # input paths, see supp_fig10_inputs.example.json
CELLS = [tuple(map(float, l.split())) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'event_mask', 'event_mask_cells.txt'))
         if l.strip() and not l.startswith('#')]   # the 36 cells (lat, lon)
IJ = [(int(round((la + 89.75) / 0.5)), int(round((lo + 179.75) / 0.5))) for la, lo in CELLS]   # south-to-north, west-to-east 0.5 deg index
out = {}
def pick(arr2d_or_more, lat_asc=True):
    return np.array([arr2d_or_more[..., i if lat_asc else 359 - i, j] for i, j in IJ])

# ---------- CRU TS 4.08 precipitation (gzip read into memory only) ----------
with gzip.open(CFG["cru_ts_pre"], "rb") as fh:
    buf = fh.read()
nc = netCDF4.Dataset("inmem", memory=buf)
lat = nc["lat"][:]; lon = nc["lon"][:]
assert abs(lat[0] + 89.75) < 1e-6 and abs(lon[0] + 179.75) < 1e-6
t = netCDF4.num2date(nc["time"][:], nc["time"].units)
yrs = np.array([d.year for d in t])
pre = nc["pre"]; stn = nc["stn"] if "stn" in nc.variables else None
res = {}
for y in range(2010, 2024):
    k = np.where(yrs == y)[0]
    blk = pre[k[0]:k[-1] + 1, :, :]
    vals = np.array([blk[:, i, j].filled(np.nan).sum() for i, j in IJ])
    s = None
    if stn is not None:
        sb = stn[k[0]:k[-1] + 1, :, :]
        s = float(np.nanmean([sb[:, i, j].astype(float).filled(np.nan).mean() for i, j in IJ]))   # stn is stored as integers
    res[y] = {"pre_mm": float(np.nanmean(vals)), "stn_mean": s, "cells_ratio_input": vals.tolist()}
out["crut408"] = res
del buf

# ---------- ERA5 tp, aligned by grid index as in the emulator pipeline ----------
e = xr.open_dataset(CFG["era5"])
tp = e["tp"]; out["era5_tp_units"] = str(tp.attrs.get("units")); out["era5_tp_long"] = str(tp.attrs.get("long_name"))
tt = e["valid_time"].to_index()
res = {}
for y in range(2010, 2025):
    k = np.where(tt.year == y)[0]
    if len(k) < 12: continue
    blk = tp.isel(valid_time=k).values                         # m per day (monthly mean of daily totals)
    dim = np.array(tt[k].days_in_month)
    vals = np.array([(blk[:, i, j] * dim).sum() * 1000.0 for i, j in IJ])
    res[y] = float(np.nanmean(vals))
out["era5"] = res

# ---------- fire and vegetation carbon in the 36 cells ----------
def annual_series(path, var, kind):
    d = xr.open_dataset(path, decode_times=False)
    v = d[var]
    tname = [x for x in v.dims if "time" in x.lower()][0]
    latn = [x for x in v.dims if x.lower().startswith("lat")][0]
    lonn = [x for x in v.dims if x.lower().startswith("lon")][0]
    latv = d[latn].values; lat_asc = latv[0] < latv[-1]
    units = d[tname].attrs["units"]; cal = d[tname].attrs.get("calendar", "standard")
    dates = cftime.num2date(d[tname].values, units, calendar=cal)
    yy = np.array([x.year for x in dates]); mm = np.array([x.month for x in dates])
    sel = (yy >= 2008) & (yy <= 2024)
    arr = v.isel({tname: np.where(sel)[0]}).transpose(tname, latn, lonn).values
    cells = np.stack([arr[:, i if lat_asc else 359 - i, j] for i, j in IJ], axis=1)   # (time, 36)
    yy, mm = yy[sel], mm[sel]
    res = {}
    n_per_year = np.bincount(yy - 2008)
    monthly = n_per_year.max() >= 12
    for y in sorted(set(yy)):
        k = yy == y
        if kind == "flux":
            if monthly:
                dim = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])[mm[k] - 1]
                val = np.nansum(cells[k] * dim[:, None] * 86400.0, axis=0) * 1000.0     # gC m-2 yr-1 (from kg m-2 s-1)
            else:
                val = cells[k][0] * 365.0 * 86400.0 * 1000.0
        else:
            val = np.nanmean(cells[k], axis=0)                                           # kg m-2 (annual or mean of months)
        res[int(y)] = float(np.nanmean(val))
    return {"series": res, "monthly": bool(monthly), "units": str(v.attrs.get("units")), "max_raw": float(np.nanmax(cells))}
F = {k: (p, k.split()[-1], "flux" if k.endswith("fFire") else "stock") for k, p in CFG["trendy"].items()}
for k, (p, v, kind) in F.items():
    try:
        out[k] = annual_series(p, v, kind)
    except Exception as ex:
        out[k] = {"error": repr(ex)}
json.dump(out, open(sys.argv[2], "w"))