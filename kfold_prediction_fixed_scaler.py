"""One-step (non-recursive) monthly prediction with a fixed normalizer.

Loads the five fold checkpoints of one experiment and predicts every month from
--start_year to --end_year, using the observed target history of the preceding
months. The RobustScaler is fitted once on --normalizer_start_year to
--normalizer_end_year (the training years) and saved, or loaded with
--normalizer_mode load, so the one-step and recursive runs use the same scaling.

Writes <result_dir>/<exp_id>.csv (five-fold mean) and <exp_id>_<fold>.csv.
These files are also the history passed to multi_prediction_evaluation.py with
--historical_result_dir. See scripts/onestep.sh.
"""
import argparse
import glob
import importlib
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from multi_prediction_evaluation import prepare_fixed_normalizer
from pipeline import Pipeline, end, setup
from process_data import get_data


def main(rank, world_size):
    setup(rank, world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--dump_dir', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--preprocessed_data_dir', required=True)
    parser.add_argument('--pft_data_dir', required=True)
    parser.add_argument('--data_name', default='data25_0616')
    parser.add_argument('--start_year', type=int, default=2010)
    parser.add_argument('--end_year', type=int, default=2023)
    parser.add_argument('--normalizer_mode', choices=['auto', 'load'], default='auto')
    parser.add_argument('--normalizer_dir', required=True)
    parser.add_argument('--normalizer_path', default='')
    parser.add_argument('--normalizer_start_year', type=int, default=2000)
    parser.add_argument('--normalizer_end_year', type=int, default=2021)
    parser.add_argument('--pft_freeze_year', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    args.window = 13
    args.patch = 18
    args.model = 'cnn'
    args.stage = 'test'
    args.device = '0'
    args.disable_tqdm = True

    os.makedirs(args.result_dir, exist_ok=True)
    pipeline = Pipeline(rank, args=args)
    data = get_data(mode=args.data_name, config=pipeline.config)
    prepare_fixed_normalizer(rank, args, pipeline, data)
    pipeline.config['input_dim'] = data.shape[1] - 1

    model_class = importlib.import_module(f'model.{pipeline.config["model"]}')
    pipeline.models = []
    for fold in range(5):
        candidates = glob.glob(
            os.path.join(
                args.dump_dir,
                str(args.exp_id),
                str(fold),
                'finish_*.pth',
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                f'Expected one finished checkpoint for target {args.exp_id} '
                f'fold {fold}, found {len(candidates)}: {candidates}'
            )
        model = model_class.Net(pipeline.config).to(pipeline.device)
        model = DDP(model, device_ids=[pipeline.rank], find_unused_parameters=True)
        model.load_state_dict(torch.load(candidates[0], map_location=pipeline.device))
        pipeline.models.append(model)

    pipeline.test_loader = pipeline.build_dataloader_all_indexes(
        data,
        'valid',
        args.start_year,
        args.end_year,
        shuffle=False,
        world_size=world_size,
    )
    pipeline.kfold_test(save_file=True)
    end()


if __name__ == '__main__':
    main(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']))
