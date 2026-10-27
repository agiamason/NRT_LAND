# Regional monthly totals for Africa (RECCAP2 ids 23-25), used for one number in steps/key_numbers.py. The regional
# analysis of the paper (ten regions) is in steps/regional_fit_for_purpose.py and steps/climatology_baseline.py.
import os
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict


EXPECTED_MODELS = [
    "CABLE-POP", "CLASSIC", "CLM6.0", "EDv3", "E3SM", "IBIS", "ISAM", "ISBA-CTRIP",
    "JSBACH", "JULES", "LPJwsl", "LPX-Bern", "OCN", "ORCHIDEE", "SDGVM", "VISIT",
    "visit-ut", "iMAPLE", "CARDAMOM"
]

RECCAP_MASK_FILE = RECCAP_MASK

BROAD_REGIONS = OrderedDict({
    "Africa": {"ids": [23, 24, 25]},
})

def _first_data_var(obj):
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(obj, xr.Dataset):
        return obj[list(obj.data_vars)[0]]
    raise TypeError(f"Unsupported object type: {type(obj)}")

def _standardize_latlon(da):
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Cannot find lat/lon dimensions in {da.dims}")
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.reindex(lat=da["lat"][::-1])
    return da

def _seconds_per_month(time_values):
    t = pd.DatetimeIndex(pd.to_datetime(time_values))
    return xr.DataArray(
        t.days_in_month.values.astype(float) * 24 * 3600,
        coords={"time": time_values},
        dims=("time",)
    )

if "model_results" not in globals() or "cnn_results" not in globals():
    raise RuntimeError("model_results and/or cnn_results are not in notebook memory.")

if not os.path.exists(RECCAP_MASK_FILE):
    raise FileNotFoundError(RECCAP_MASK_FILE)

reccap_ds = xr.open_dataset(RECCAP_MASK_FILE)
reccap = _standardize_latlon(_first_data_var(reccap_ds))

available_models = [m for m in EXPECTED_MODELS if m in model_results and m in cnn_results]
if not available_models:
    raise RuntimeError("No expected target models found in model_results/cnn_results.")

ref0 = _standardize_latlon(_first_data_var(model_results[available_models[0]]))

# Align RECCAP mask to target-model grid if needed.
if (
    not np.array_equal(reccap["lat"].values, ref0["lat"].values)
    or not np.array_equal(reccap["lon"].values, ref0["lon"].values)
):
    reccap = reccap.interp(lat=ref0["lat"], lon=ref0["lon"], method="nearest")

# grid-cell area as in cal_mon
lat = ref0["lat"].values
lon = ref0["lon"].values
area_2d = xr.DataArray(seaflux.area.area_grid(lat, lon, return_dataarray=True).values,
                       coords={"lat": ref0["lat"], "lon": ref0["lon"]}, dims=("lat", "lon"))

def _build_region_mask(spec):
    land = np.isfinite(reccap) & (reccap > 0)

    if spec.get("kind") == "lat":
        cond = land & (reccap["lat"] >= spec["lat_min"]) & (reccap["lat"] <= spec["lat_max"])
    else:
        ids = spec["ids"]
        cond = xr.zeros_like(reccap, dtype=bool)
        for rid in ids:
            cond = cond | (reccap == rid)
        if "lat_min" in spec:
            cond = cond & (reccap["lat"] >= spec["lat_min"]) & (reccap["lat"] <= spec["lat_max"])

    return xr.where(cond, 1.0, np.nan)

region_masks = OrderedDict()
region_inventory = []

for region_name, spec in BROAD_REGIONS.items():
    mask = _build_region_mask(spec)
    region_masks[region_name] = mask
    area_mkm2 = float((area_2d * mask).sum(skipna=True).values / 1e12)
    n_cells = int(np.isfinite(mask.values).sum())

    if spec.get("kind") == "lat":
        definition = f"land mask, latitude {spec['lat_min']} to {spec['lat_max']}"
    else:
        definition = f"RECCAP IDs {spec['ids']}"
        if "lat_min" in spec:
            definition += f", latitude {spec['lat_min']} to {spec['lat_max']}"

    region_inventory.append({
        "Region": region_name,
        "Definition": definition,
        "N_grid_cells": n_cells,
        "Area_10^6_km2": area_mkm2,
    })

region_inventory_df = pd.DataFrame(region_inventory)

print("\n### BEGIN_TABLE: broad_region_inventory")
print(region_inventory_df.to_csv(sep="\t", index=False))
print("### END_TABLE: broad_region_inventory")

def _regional_monthly_gtc(da, mask):
    da = _standardize_latlon(_first_data_var(da))
    sec = _seconds_per_month(da["time"].values)
    # Assumes monthly NBP flux unit kgC m-2 s-1. Output GtC month-1.
    return (da * sec * area_2d * mask / 1e12).sum(dim=("lat", "lon"), skipna=True)
