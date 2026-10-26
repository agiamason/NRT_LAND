import glob
import importlib
import argparse
import json
import os
import pickle
import subprocess
import torch
import pandas as pd
import numpy as np
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


def month_from_key(key):
    """Convert the integer month key used by :func:`month_key` back to Y/M."""
    year, month = divmod(int(key), 12)
    if month == 0:
        year -= 1
        month = 12
    return year, month


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


def initialize_outputs_from_historical(rank, args):
    """Clone historical CSVs and remove rows at/after the recursive start year.

    With --historical_result_dir, the output CSVs start as copies of the one-step
    (non-recursive) result files of the same experiment, cut just before
    --start_year, and the recursive months are appended to them. The copy uses
    GNU cp/grep, as on the Linux systems the code was run on.
    """
    if not args.historical_result_dir:
        return

    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        suffixes = ['', '_0', '_1', '_2', '_3', '_4']
        for suffix in suffixes:
            source = os.path.join(
                args.historical_result_dir, f'{args.exp_id}{suffix}.csv'
            )
            destination = os.path.join(args.result_dir, f'{args.exp_id}{suffix}.csv')
            if not os.path.exists(source):
                raise FileNotFoundError(f'historical result not found: {source}')

            # A reflink makes this nearly instantaneous on filesystems that
            # support copy-on-write and safely falls back to a normal copy.
            subprocess.run(
                ['cp', '--reflink=auto', '--', source, destination],
                check=True,
            )
            match = subprocess.run(
                [
                    'grep', '-E', '-b', '-m', '1',
                    f'^{args.start_year}(\\.0)?,',
                    destination,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if match.returncode == 0:
                byte_offset = int(match.stdout.split(':', 1)[0])
                os.truncate(destination, byte_offset)
                print(
                    f'initialized {destination} from {source}; '
                    f'truncated before {args.start_year} at byte {byte_offset}'
                )
            elif match.returncode == 1:
                # Training-time result files commonly end at the year just
                # before recursive prediction begins.  In that valid case
                # there is nothing to truncate.
                with open(destination, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    pos = f.tell() - 1
                    while pos >= 0:
                        f.seek(pos)
                        if f.read(1) not in (b'\n', b'\r'):
                            break
                        pos -= 1
                    end_pos = pos
                    while pos >= 0:
                        f.seek(pos)
                        if f.read(1) == b'\n':
                            break
                        pos -= 1
                    f.seek(pos + 1)
                    last_line = f.read(end_pos - pos).decode('utf-8').strip()
                last_year = int(float(last_line.split(',', 1)[0]))
                if last_year >= args.start_year:
                    raise ValueError(
                        f'Could not locate year {args.start_year} in {destination}; '
                        f'last observed year is {last_year}'
                    )
                print(
                    f'initialized {destination} from {source}; historical '
                    f'file already ends at {last_year}, so no truncation was needed'
                )
            else:
                raise RuntimeError(
                    f'grep failed while preparing {destination}: {match.stderr}'
                )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def apply_source_missing_mask(result, source_month, year, month):
    """Apply only the source target's missingness mask, aligned by coordinates."""
    frame = result.reset_index() if isinstance(result.index, pd.MultiIndex) else result.copy()
    frame = frame.merge(
        source_month,
        on=['lat', 'lon'],
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if frame['_target_is_nan'].isna().any():
        missing = int(frame['_target_is_nan'].isna().sum())
        raise ValueError(
            f'Could not align {missing} mask coordinates for {year}-{month:02d}'
        )
    frame.loc[frame['_target_is_nan'], 'predictions'] = np.nan
    return frame.drop(columns=['_target_is_nan'])


def append_month_outputs(rank, args, mean_result, fold_results, first_month):
    if rank != 0:
        return
    # Without --historical_result_dir, new files (with a header) are started at
    # the first prediction month; otherwise rows are appended to the copies.
    fresh = first_month and not args.historical_result_dir
    mode, header = ('w', True) if fresh else ('a', False)
    os.makedirs(args.result_dir, exist_ok=True)
    for fold_index, fold_result in enumerate(fold_results):
        path = os.path.join(args.result_dir, f'{args.exp_id}_{fold_index}.csv')
        fold_result.to_csv(path, mode=mode, header=header, index=False)
    mean_path = os.path.join(args.result_dir, f'{args.exp_id}.csv')
    mean_result.to_csv(mean_path, mode=mode, header=header, index=False)


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
    parser.add_argument('--pft_freeze_year', type=int, default=0,
                        help='hold PFT fractions at this year for all later years (0 = off)')
    parser.add_argument('--mask_year', type=int, default=0,
                        help='months after this year use the missing-data mask of the same month of this year '
                             '(needed when the target has no values in the prediction year; 0 = off)')
    parser.add_argument('--data_name', type=str, help='data name', default='2026_5')
    parser.add_argument('--start_year', type=int, help='start year', default=2025)
    parser.add_argument('--start_month', type=int, help='start month', default=1)
    parser.add_argument('--end_year', type=int, help='end year', default=2026)
    parser.add_argument('--end_month', type=int, help='end month', default=3)
    parser.add_argument('--normalizer_mode', type=str, choices=['auto', 'load', 'off'], default='auto')
    parser.add_argument('--normalizer_dir', type=str, default='./normalizers')
    parser.add_argument('--normalizer_path', type=str, default='')
    parser.add_argument('--normalizer_start_year', type=int, default=2000)
    parser.add_argument('--normalizer_end_year', type=int, default=2024)
    parser.add_argument('--historical_result_dir', type=str, default='',
                        help='one-step result directory whose CSVs are copied and continued (optional)')
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
    initialize_outputs_from_historical(rank, args)
    
    mt.config['input_dim'] = data.shape[1] - 1
    print('input_dim', mt.config['input_dim'])
    # The land mask is taken from the target column already present in the
    # evaluation feather, month by month.  Only missingness is read; target
    # values are never used as model input for the prediction month.

    model_paths = sorted(
        glob.glob(mt.config['dump_dir'] + f'/*{mt.config["exp_id"]}/*/finish_*'),
        key=lambda path: int(os.path.basename(os.path.dirname(path))),
    )
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
            # Build a loader only for the 13-month window ending at this
            # prediction month.
            target_key = month_key(year, month)
            history_start_year, history_start_month = month_from_key(
                target_key - (mt.config['window'] - 1)
            )
            history_start_key = month_key(history_start_year, history_start_month)
            data_keys = data['year'].astype(np.int32) * 12 + data['month'].astype(np.int32)
            month_data = data.loc[
                (data_keys >= history_start_key) & (data_keys <= target_key)
            ].copy()
            expected_rows = 360 * 720 * mt.config['window']
            if month_data.shape[0] != expected_rows:
                raise ValueError(
                    f'Incomplete evaluation window for {year}-{month:02d}: '
                    f'{history_start_year}-{history_start_month:02d}..{year}-{month:02d} '
                    f'has {month_data.shape[0]} rows, expected {expected_rows}'
                )
            print(
                f'evaluation window: {history_start_year}-{history_start_month:02d}..'
                f'{year}-{month:02d} ({month_data.shape[0]} rows)'
            )
            mt.test_loader = mt.build_dataloader_all_indexes(
                month_data,
                'valid',
                history_start_year,
                year,
                shuffle=False,
                world_size=world_size,
            )

            # Get all five fold results for this month so that complete fold
            # and ensemble CSVs can be written.
            result, fold_results = mt.kfold_test(
                save_file=False,
                return_fold_results=True,
            )
            result = result.reset_index()
            cur_res = result[(result.year == year) & (result.month == month)].copy()

            # Preserve the original target's land/missing-data mask without
            # reading any true target values.  Join by coordinates so masking
            # and recursive updates do not depend on incidental row order.
            # In a year without target values (2024 in the paper's second
            # experiment) every prediction would be masked to NaN and the NaN fed
            # back as history. With --mask_year Y, months after Y use the
            # missing-data mask of the same month of Y.
            mask_year = args.mask_year if (args.mask_year and year > args.mask_year) else year
            source_month = data.loc[
                (data.year == mask_year) & (data.month == month),
                ['latitude', 'longitude', mt.config['target']],
            ].copy()
            source_month = source_month.rename(
                columns={'latitude': 'lat', 'longitude': 'lon'}
            )
            source_month['_target_is_nan'] = source_month[mt.config['target']].isna()
            source_month = source_month[['lat', 'lon', '_target_is_nan']]
            if source_month.duplicated(['lat', 'lon']).any():
                raise ValueError(f'Duplicate source coordinates for {year}-{month:02d}')

            cur_res = apply_source_missing_mask(
                cur_res, source_month, year, month
            )
            masked_folds = []
            for fold_result in fold_results:
                fold_month = fold_result.reset_index()
                fold_month = fold_month[
                    (fold_month.year == year) & (fold_month.month == month)
                ].copy()
                masked_folds.append(
                    apply_source_missing_mask(
                        fold_month, source_month, year, month
                    )
                )
            append_month_outputs(
                rank, args, cur_res, masked_folds,
                first_month=(year == args.start_year and month == args.start_month),
            )

            # Feed this prediction back as history for the next month.  This is
            # the only recursive target update; no target values from the
            # prediction period are inserted into history.
            base = data.loc[
                (data.year == year) & (data.month == month),
                ['latitude', 'longitude'],
            ].reset_index()
            update = base.merge(
                cur_res[['lat', 'lon', 'predictions']],
                left_on=['latitude', 'longitude'],
                right_on=['lat', 'lon'],
                how='left',
                sort=False,
                validate='one_to_one',
            )
            if update['predictions'].isna().sum() != cur_res['predictions'].isna().sum():
                raise ValueError(
                    f'Prediction-coordinate alignment changed the NaN count for '
                    f'{year}-{month:02d}'
                )
            data.loc[
                update['index'].to_numpy(), mt.config['target']
            ] = update['predictions'].to_numpy() * 1e-10
            

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

