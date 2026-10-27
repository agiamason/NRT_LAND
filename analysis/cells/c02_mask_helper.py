# Helper: 1 where a field has data, NaN elsewhere.
def mask_a(mask: xr.DataArray) -> xr.DataArray:
    """Return a field that is 1 where `mask` has data and NaN elsewhere (same coordinates)."""
    mask_binary = xr.DataArray(np.where(np.isnan(mask), np.nan, 1), coords=mask.coords, dims=mask.dims)

    return mask_binary
