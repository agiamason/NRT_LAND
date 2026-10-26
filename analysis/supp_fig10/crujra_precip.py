"""Supp Fig 10, panel a (CRU-JRA): annual precipitation (Rainf + Snowf, mm yr-1) of CRU-JRA v2.3 (2010-2021) and v2.4
(2010-2022) in each of the 36 excluded cells, from the half-degree ORCHIDEE forcing files (6-hourly, land points only).
Also the area-weighted mean over the 10-degree box 0-10N, 80-70W, and the v2.5 10-degree test files if present.

    python crujra_precip.py CRUJRA_DIR crujra_precip.json
"""
import warnings; warnings.filterwarnings("ignore")
import os, xarray as xr, numpy as np, json, sys
B = os.path.join(sys.argv[1], "")   # directory with v2.3/halfdeg, v2.4/halfdeg (and v2.5/tendeg)
CELLS = [tuple(map(float, l.split())) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'event_mask', 'event_mask_cells.txt'))
         if l.strip() and not l.startswith('#')]   # the 36 cells (lat, lon)
out = {}
def halfdeg(ver, years):
    res = {"cells": {}, "box10": {}}
    idx = None
    for y in years:
        d = xr.open_dataset(f"{B}{ver}/halfdeg/crujra_halfdeg_{ver}_{y}.nc", decode_times=False)
        if idx is None:
            land = d["land"].values.astype(int) - 1           # 1-based index into the y*x grid
            nl, nx = d.sizes["y"], d.sizes["x"]
            lat2, lon2 = d["nav_lat"].values, d["nav_lon"].values
            lat_l, lon_l = lat2.ravel()[land], lon2.ravel()[land]
            idx = []
            for la, lo in CELLS:
                k = np.where((np.abs(lat_l - la) < 0.01) & (np.abs(lon_l - lo) < 0.01))[0]
                idx.append(int(k[0]) if len(k) else -1)
            box = np.where((lat_l > 0) & (lat_l < 10) & (lon_l > -80) & (lon_l < -70))[0]
            wbox = np.cos(np.deg2rad(lat_l[box]))
        nt = d.sizes["tstep"]
        pr = (d["Rainf"].values + d["Snowf"].values)               # kg m-2 s-1, (tstep, land)
        ann = np.nansum(pr, axis=0) * 21600.0                       # mm per year (6-hourly steps)
        res["cells"][y] = [float(ann[i]) if i >= 0 else None for i in idx]
        res["box10"][y] = float(np.nansum(ann[box] * wbox) / np.nansum(wbox))
        res.setdefault("tsteps", {})[y] = int(nt)
    res["missing_cells"] = int(sum(1 for i in idx if i < 0))
    return res
out["v2.3"] = halfdeg("v2.3", range(2010, 2022))
out["v2.4"] = halfdeg("v2.4", range(2010, 2023))
# v2.5: 10-degree test files only; the 10x10 box 0-10N, 80-70W (for the caption; skipped if absent)
r = {}
for y in (range(2010, 2025) if os.path.isdir(B + "v2.5/tendeg") else []):
    d = xr.open_dataset(f"{B}v2.5/tendeg/crujra_tendeg_v2.5_{y}.nc", decode_times=False)
    la, lo = d["nav_lat"].values, d["nav_lon"].values
    if la.ndim == 1: la, lo = np.meshgrid(la, lo, indexing="ij")
    j = np.where((la > 0) & (la < 10) & (lo > -80) & (lo < -70))
    pr = (d["Rainf"].values + d["Snowf"].values)
    ann = np.nansum(pr, axis=0) * 21600.0
    r[y] = float(ann[j][0]); r.setdefault("_cell", [float(la[j][0]), float(lo[j][0])])
out["v2.5_tendeg"] = r
json.dump(out, open(sys.argv[2], "w"))