# Evaluation mask of each model = valid mask (complete over 2010-2023 and temporal sd > 1e-10 kg m-2 s-1) minus 36
# land cells in western Venezuela where the CRU-based forcing of the TRENDY models shows a decrease in 2021
# precipitation that is absent from ERA5 (EVENT_MASK, see analysis/event_mask/). Built from model output and forcing only, never from emulator output. Applied to the model and
# the emulator fields; saved to OUT_DIR/valid_mask_final.nc.
ev = xr.open_dataset(EVENT_MASK)
_ref = model_results[list(DGVM_list.values())[0]]
assert np.allclose(ev['lat'].values, _ref['lat'].values) and np.allclose(ev['lon'].values, _ref['lon'].values)
event_mask = xr.DataArray(ev['event_mask'].values.astype(bool), coords={'lat': _ref['lat'], 'lon': _ref['lon']}, dims=('lat', 'lon'))
print(f"excluded cells of western Venezuela (event mask): {int(event_mask.sum())}")

valid_masks, final_masks, mask_rows = {}, {}, []
for dgvm_name in DGVM_list.values():
    vm = valid_mask(model_results[dgvm_name])
    fm = vm & ~event_mask
    valid_masks[dgvm_name], final_masks[dgvm_name] = vm, fm
    model_results[dgvm_name] = model_results[dgvm_name].where(fm)
    cnn_results[dgvm_name] = cnn_results[dgvm_name].where(fm)
    mask_rows.append({'Model': disp(dgvm_name), 'n_valid': int(vm.sum()),
                      'n_removed_event': int((vm & event_mask).sum()), 'n_final': int(fm.sum())})
mask_table = pd.DataFrame(mask_rows).set_index('Model')
print(mask_table.to_string())

_names = list(DGVM_list.values())
ds_mask = xr.Dataset(
    {'valid_mask_final': (('DGVM', 'lat', 'lon'), np.stack([final_masks[n].values for n in _names]).astype(np.int8)),
     'valid_mask': (('DGVM', 'lat', 'lon'), np.stack([valid_masks[n].values for n in _names]).astype(np.int8)),
     'event_mask': (('lat', 'lon'), event_mask.values.astype(np.int8))},
    coords={'DGVM': _names, 'lat': _ref['lat'].values, 'lon': _ref['lon'].values})
ds_mask['valid_mask_final'].attrs['definition'] = '1 = used in all figures, numbers and tables: valid_mask AND NOT event_mask'
ds_mask['valid_mask'].attrs['definition'] = 'TRENDY v13 S2 nbp complete over 2010-01..2023-12 and temporal standard deviation > 1e-10 kg m-2 s-1'
ds_mask['event_mask'].attrs['definition'] = ev['event_mask'].attrs.get('definition', '')
ds_mask.attrs.update({'source_model': os.path.basename(TRENDY_V13_NBP),
                      'source_event_mask': ev.attrs.get('source', ''),
                      'note': 'DGVM names as in the TRENDY v13 file (visit-ut = VISIT-UT)',
                      'created_by': 'analysis/cells/c06_evaluation_mask.py'})
ds_mask.to_netcdf(OUT_DIR + 'valid_mask_final.nc')
print('Saved', OUT_DIR + 'valid_mask_final.nc')
