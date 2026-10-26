# Fig 4, part 1: functions (also used by steps/model_differences.py).
apply_style()

def get_truncated_purples():
    cmap = plt.get_cmap('Purples')
    minval = 0.5  # darker start of the colour map
    maxval = 1.0
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, 256)))
    return new_cmap

strict_purple_cmap = get_truncated_purples()

def scatter_component_pairs(dgvm_name):
    """Four components in gC m-2 (month-1 or yr-1). Day-accurate units; mean seasonal cycle from the
    2022-2023 test period (sc_model / sc_cnn); clim_model / clim_cnn stay 2010-2021 as the anomaly base."""
    raw_model, raw_cnn = model_results[dgvm_name], cnn_results[dgvm_name]
    da_model, da_cnn = to_gC_month(raw_model), to_gC_month(raw_cnn)
    # [Col 1] Monthly NBP (Test Period)
    x1 = da_model.sel(time=test_slice).values.flatten()
    y1 = da_cnn.sel(time=test_slice).values.flatten()
    # [Col 2] Mean Annual / Spatial (Test Period), mean of the calendar-year totals
    x2 = annual_gC(raw_model.sel(time=test_slice)).mean('year').values.flatten()
    y2 = annual_gC(raw_cnn.sel(time=test_slice)).mean('year').values.flatten()
    # [Col 3] Mean Seasonal Cycle (Test Period 2022-2023)
    sc_model = da_model.sel(time=test_slice).groupby('time.month').mean('time')
    sc_cnn = da_cnn.sel(time=test_slice).groupby('time.month').mean('time')
    x3 = sc_model.values.flatten()
    y3 = sc_cnn.values.flatten()
    # [Col 4] Monthly Anomalies (Test Period), anomaly base = own 2010-2021 climatology
    clim_model = da_model.sel(time=ref_slice).groupby('time.month').mean('time')
    clim_cnn = da_cnn.sel(time=ref_slice).groupby('time.month').mean('time')
    anom_model = calculate_anomaly(da_model.sel(time=test_slice), clim_model)
    anom_cnn = calculate_anomaly(da_cnn.sel(time=test_slice), clim_cnn)
    x4 = anom_model.values.flatten()
    y4 = anom_cnn.values.flatten()
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
