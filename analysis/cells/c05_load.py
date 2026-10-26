# Load TRENDY v13 S2 monthly NBP (kg m-2 s-1, 2010-2023) and the emulator output of experiment 1 (one CSV per model,
# rows ordered by year, month, lat, lon; the emulator writes NBP x 1e10). Only the 'predictions' column is read.
cnn_path = EMULATOR_EXP1_DIR   # experiment 1: one-step rows 2010-2021, recursive 2022-2023

model_dataset = xr.open_dataset(TRENDY_V13_NBP)
model_dataset = model_dataset.sel(time=slice('2010-01-01', '2023-12-31'))

model_results = {}
cnn_results = {}

for dgvm_id, dgvm_name in DGVM_list.items():
    cnn_df = pd.read_csv(cnn_path + f'{dgvm_id}.csv', engine='pyarrow', usecols=['year', 'month', 'predictions'])

    cnn_df = cnn_df[(cnn_df['year'] >= 2010) & (cnn_df['year'] < 2024) ]
    cnn = cnn_df['predictions'].values / (1e10)

    model = model_dataset.sel(DGVM=dgvm_name).nbp
    model_shape = model.shape  # (time, lat, lon)
    model_time = pd.date_range(start='2009-12-15', periods=model_shape[0], freq='MS') + pd.DateOffset(days=14)
    model.coords['time'] = model_time

    total_months = model_shape[0]
    cnn = cnn.reshape((total_months, *model_shape[1:]))

    # emulator output is masked where the model has no data
    cnn[:168][np.isnan(model.values)] = np.nan

    time_index = pd.date_range(start='2009-12-15', periods=total_months, freq='MS') + pd.DateOffset(days=14)

    cnn_xr = xr.DataArray(cnn, dims=['time'] + list(model.dims[1:]),
                          coords={'time': time_index, 'lat': model.coords['lat'], 'lon': model.coords['lon']})
    cnn_xr.attrs = model.attrs

    model_results[dgvm_name] = model
    cnn_results[dgvm_name] = cnn_xr
