# Fig 3, part 1: metric functions (also used by later steps).
apply_style()

def calc_pixel_wise_metrics(da_obs, da_pred, time_dim='time'):
    """Method 1: R2 (squared Pearson) per pixel, averaged over valid pixels
    (temporal sd of the reference > 1e-10). Also returns the median over pixels, the pooled R2 over all
    valid pixel-time samples and the median of the per-pixel RMSE (shown in version ii)."""
    std_map = da_obs.std(dim=time_dim)
    valid_px = std_map > 1e-10
    r2_map = xr.corr(da_obs, da_pred, dim=time_dim) ** 2
    rmse_map = np.sqrt(((da_obs - da_pred) ** 2).mean(dim=time_dim))
    return {'r2_mean': float(r2_map.where(valid_px).mean()),
            'r2_median': float(r2_map.where(valid_px).median()),
            'r2_pooled': pearson_r2(da_obs.where(valid_px).values, da_pred.where(valid_px).values),
            'rmse_median': float(rmse_map.where(valid_px).median())}

def spatial_metrics(map_obs, map_pred):
    x = map_obs.values.ravel(); y = map_pred.values.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return {'r2': pearson_r2(x[ok], y[ok]), 'rmse': float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))}
