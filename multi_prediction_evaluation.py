import glob
import importlib
import argparse
import json
import os
import pickle
import torch
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import RobustScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from process_data import  get_data
from pipeline import Pipeline, setup, end

def normalizer_paths(args):
    if args.normalizer_path:
        pkl_path = args.normalizer_path
        json_path = os.path.splitext(pkl_path)[0] + '.json'
    else:
        base = (
            f'normalizer_land_{args.target}_{args.data_name}_'
            f'{args.normalizer_start_year}_{args.normalizer_end_year}'
        )
        pkl_path = os.path.join(args.normalizer_dir, f'{base}.pkl')
        json_path = os.path.join(args.normalizer_dir, f'{base}.json')
    return pkl_path, json_path


def fit_fixed_normalizer(data, norm_columns, args):
    fit_data = data[
        (data['year'] >= args.normalizer_start_year)
        & (data['year'] <= args.normalizer_end_year)
    ].copy()
    if fit_data.empty:
        raise ValueError(
            f'No rows found for normalizer fit range '
            f'{args.normalizer_start_year}-{args.normalizer_end_year}'
        )

    target_scale = 1.0
    target_mean = fit_data[args.target].mean()
    if np.abs(target_mean) < 1e-6:
        target_scale = 1e10
        fit_data[args.target] = fit_data[args.target] * target_scale

    normalizer = {}
    summary = {
        'target': args.target,
        'data_name': args.data_name,
        'start_year': args.normalizer_start_year,
        'end_year': args.normalizer_end_year,
        'norm_columns': norm_columns,
        'rows': int(len(fit_data)),
        'target_raw_mean': float(target_mean),
        'target_scale_before_normalize': float(target_scale),
        'columns': {},
    }

    for col in norm_columns:
        if col not in fit_data.columns:
            raise KeyError(f'Missing normalizer column: {col}')
        values = fit_data[col].values.reshape(-1, 1)
        finite_count = int(np.isfinite(values).sum())
        if finite_count == 0:
            raise ValueError(f'Column {col} has no finite values in normalizer fit range')
        scaler = RobustScaler()
        scaler.fit(values)
        normalizer[col] = scaler
        summary['columns'][col] = {
            'finite_count': finite_count,
            'nan_count': int(np.isnan(values).sum()),
            'center': float(scaler.center_[0]),
            'scale': float(scaler.scale_[0]),
        }

    return normalizer, summary


def prepare_fixed_normalizer(rank, args, mt, data):
    if args.normalizer_mode == 'off':
        if rank == 0:
            print('fixed normalizer disabled; Pipeline will fit on prediction range')
        return

    pkl_path, json_path = normalizer_paths(args)
    needs_fit = not os.path.exists(pkl_path)

    if needs_fit and args.normalizer_mode == 'load':
        raise FileNotFoundError(f'normalizer not found: {pkl_path}')

    if rank == 0 and needs_fit:
        output_dir = os.path.dirname(pkl_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        normalizer, summary = fit_fixed_normalizer(data, mt.config['norm_columns'], args)
        with open(pkl_path, 'wb') as f:
            pickle.dump(normalizer, f)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'fitted fixed normalizer: {pkl_path}')
        print(f'fixed normalizer report: {json_path}')

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    with open(pkl_path, 'rb') as f:
        mt.normalizer = pickle.load(f)
    if rank == 0:
        print(f'loaded fixed normalizer: {pkl_path}')


def month_key(year, month):
    return int(year) * 12 + int(month)


def iter_months(start_year, start_month, end_year, end_month):
    cur = month_key(start_year, start_month)
    end_key = month_key(end_year, end_month)
    while cur <= end_key:
        year = cur // 12
        month = cur % 12
        if month == 0:
            year -= 1
            month = 12
        yield year, month
        cur += 1


def validate_month_range(start_year, start_month, end_year, end_month, label):
    if not 1 <= start_month <= 12:
        raise ValueError(f'{label} start_month must be in 1..12, got {start_month}')
    if not 1 <= end_month <= 12:
        raise ValueError(f'{label} end_month must be in 1..12, got {end_month}')
    if month_key(start_year, start_month) > month_key(end_year, end_month):
        raise ValueError(
            f'{label} start {start_year}-{start_month:02d} is after '
            f'end {end_year}-{end_month:02d}'
        )


def load_warm_start_predictions(csv_path, args):
    usecols = ['year', 'month', 'lat', 'lon', 'predictions']
    dtype = {
        'year': 'int16',
        'month': 'int8',
        'lat': 'float32',
        'lon': 'float32',
        'predictions': 'float32',
    }
    start_key = month_key(args.warm_start_start_year, args.warm_start_start_month)
    end_key = month_key(args.warm_start_end_year, args.warm_start_end_month)
    chunks = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=args.warm_start_chunksize,
    ):
        keys = chunk['year'].astype(np.int32) * 12 + chunk['month'].astype(np.int32)
        chunk = chunk[(keys >= start_key) & (keys <= end_key)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f'No warm-start rows found in {csv_path} for '
            f'{args.warm_start_start_year}-{args.warm_start_start_month:02d}..'
            f'{args.warm_start_end_year}-{args.warm_start_end_month:02d}'
        )

    warm = pd.concat(chunks, ignore_index=True)
    warm.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
    return warm


def apply_warm_start_predictions(rank, args, data):
    if not args.warm_start_result_dir:
        return

    validate_month_range(
        args.warm_start_start_year,
        args.warm_start_start_month,
        args.warm_start_end_year,
        args.warm_start_end_month,
        'warm_start',
    )

    csv_path = os.path.join(args.warm_start_result_dir, f'{args.exp_id}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'warm-start result not found: {csv_path}')

    if rank == 0:
        print(
            f'loading warm-start predictions: {csv_path} '
            f'({args.warm_start_start_year}-{args.warm_start_start_month:02d}..'
            f'{args.warm_start_end_year}-{args.warm_start_end_month:02d})'
        )

    warm = load_warm_start_predictions(csv_path, args)

    for year, month in iter_months(
        args.warm_start_start_year,
        args.warm_start_start_month,
        args.warm_start_end_year,
        args.warm_start_end_month,
    ):
        idx = data[(data.year == year) & (data.month == month)].index
        if idx.empty:
            raise ValueError(f'No data rows found for warm-start month {year}-{month:02d}')

        pred_month = warm[(warm.year == year) & (warm.month == month)][
            ['latitude', 'longitude', 'predictions']
        ]
        if pred_month.empty:
            raise ValueError(f'No prediction rows found for warm-start month {year}-{month:02d}')

        base = data.loc[idx, ['latitude', 'longitude']].reset_index()
        merged = base.merge(
            pred_month,
            on=['latitude', 'longitude'],
            how='left',
            sort=False,
            indicator=True,
        )
        missing = int((merged['_merge'] != 'both').sum())
        if missing:
            raise ValueError(
                f'warm-start month {year}-{month:02d} missing {missing} '
                f'lat/lon prediction rows'
            )

        values = merged['predictions'].to_numpy(dtype=np.float32) * args.warm_start_scale
        data.loc[merged['index'].to_numpy(), args.target] = values
        if rank == 0:
            finite_count = int(np.isfinite(values).sum())
            print(
                f'warm-start applied {year}-{month:02d}: '
                f'rows={len(values)}, finite={finite_count}'
            )


def main(rank, world_size):
    setup(rank, world_size)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, help='exp id', default=201)
    parser.add_argument('--target', type=str, help='model name', default='ORCHIDEE_nbp')
    parser.add_argument('--dump_dir', type=str, help='model dir', default='./checkpoint/')
    parser.add_argument('--result_dir', type=str, help='result dir', default='./result/')
    parser.add_argument('--preprocessed_data_dir', type=str, help='preprocessed_data_dir', default='./data/')
    parser.add_argument('--pft_data_dir', type=str, help='pft_data_dir', default='./data/PFT/')
    parser.add_argument('--data_name', type=str, help='data name', default='2026_5')
    parser.add_argument('--dgvm_path', type=str, help='dgvm_path', default='./data/Trendyv14_S2_nbp.nc')
    parser.add_argument('--dgvm_average_path', '--dgvm_mean_path', dest='dgvm_average_path', type=str, help='dgvm_average_path', default='./data/average_nbp.nc')
    parser.add_argument('--start_year', type=int, help='start year', default=2025)
    parser.add_argument('--start_month', type=int, help='start month', default=1)
    parser.add_argument('--end_year', type=int, help='end year', default=2026)
    parser.add_argument('--end_month', type=int, help='end month', default=3)
    parser.add_argument('--test_start_year', type=int, help='test dataloader start year', default=2000)
    parser.add_argument('--test_end_year', type=int, help='test dataloader end year; defaults to end_year', default=None)
    parser.add_argument('--normalizer_mode', type=str, choices=['auto', 'load', 'off'], default='auto')
    parser.add_argument('--normalizer_dir', type=str, default='./normalizers')
    parser.add_argument('--normalizer_path', type=str, default='')
    parser.add_argument('--normalizer_start_year', type=int, default=2000)
    parser.add_argument('--normalizer_end_year', type=int, default=2024)
    parser.add_argument('--warm_start_result_dir', type=str, default='')
    parser.add_argument('--warm_start_start_year', type=int, default=2025)
    parser.add_argument('--warm_start_start_month', type=int, default=1)
    parser.add_argument('--warm_start_end_year', type=int, default=2025)
    parser.add_argument('--warm_start_end_month', type=int, default=8)
    parser.add_argument('--warm_start_scale', type=float, default=1e-10)
    parser.add_argument('--warm_start_chunksize', type=int, default=2000000)
    args = parser.parse_args()
    validate_month_range(args.start_year, args.start_month, args.end_year, args.end_month, 'prediction')

    target = args.target
    exp_id = args.exp_id
    print(target, exp_id)
    args.window = 13 
    args.patch = 18 
    args.model = 'cnn'  
    args.stage = 'test' 

    args.target =target
    args.exp_id =exp_id
    args.device = "1"
    mt = Pipeline(rank,args=args)  # supervised/semi_supervised
    data = get_data(mode=args.data_name, config=mt.config)
    prepare_fixed_normalizer(rank, args, mt, data)
    apply_warm_start_predictions(rank, args, data)
    
    mt.config['input_dim'] = data.shape[1] - 1
    print('input_dim', mt.config['input_dim'])
    if target == 'average_nbp':
        dgvm_path =  args.dgvm_average_path
        pro_data = xr.open_dataset(dgvm_path)
        last_12_months_nan = np.isnan(pro_data['nbp'][-12:,:,:].values)
    else:
        dgvm_path = args.dgvm_path
    
        pro_data = xr.open_dataset(dgvm_path)

        for idx, nbp_name in enumerate(pro_data['DGVM'][:].values):
            if nbp_name not in target:
                continue
            print(idx, nbp_name)
            last_12_months_nan = np.isnan(pro_data['nbp'][idx][-12:,:,:].values)


    model_paths = glob.glob(mt.config['dump_dir'] + f'/*{mt.config["exp_id"]}/*/finish_*')
    print(model_paths)
    model_class = importlib.import_module(
        f'model.{mt.config["model"]}'
    )
    mt.models = []
    for model_path in model_paths:
        model = model_class.Net(mt.config).to(mt.device)
        model = DDP(model, device_ids=[mt.rank], find_unused_parameters=True)
        model.load_state_dict(torch.load(model_path))
        mt.models.append(model)
    print('load model finished...')
    for year in range(args.start_year, args.end_year + 1):
        first_month = args.start_month if year == args.start_year else 1
        last_month = args.end_month if year == args.end_year else 12
        for month in range(first_month, last_month + 1):
            print(year, month)
            test_start_year = args.test_start_year
            test_end_year = args.test_end_year if args.test_end_year is not None else args.end_year
            mt.test_loader = mt.build_dataloader_all_indexes(
                data,
                'valid',
                test_start_year,
                test_end_year,
                shuffle=False,
                world_size=world_size,
            )

            if year==args.end_year and month == args.end_month:
                result = mt.kfold_test()
                break
            else:
                result = mt.kfold_test(save_file=False)
            result = result.reset_index()
            cur_res = result[(result.year==year) & (result.month==month)]
            nan_mask = last_12_months_nan[month-1].flatten()
            cur_res.loc[nan_mask, 'predictions'] = np.nan

            # renew data
            idx = data[(data.year==year) & (data.month==month)].index
            data.loc[idx, mt.config['target']] = cur_res.predictions.values * 1e-10
            

    print('iterative prediction finished!')

    end()


from torch.utils.data import DistributedSampler
import torch.distributed as dist
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print('world_size1:', world_size)
    world_size = int(os.environ['WORLD_SIZE'])
    print('world_size2:', world_size)
    rank = int(os.environ['RANK'])
    main(rank, world_size)
