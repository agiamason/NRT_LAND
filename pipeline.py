from sklearn.model_selection import KFold
from ast import parse
import os
import importlib
import argparse
import numpy as np
import pandas as pd
import data_utils
import copy
import glob

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from itertools import product
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from torch.utils.data import  DataLoader

from process_data import get_data, OceanDataset_V2
from utils import mse_loss_with_nan, ramp_up, augment, likelihood_with_mask, kl_div_with_mask, quantile_aug, StepLRWithMinLRScheduler, WarmUpLR
import time
import gc

from torch.optim.lr_scheduler import CosineAnnealingLR

def gather_tensors(tensor, device):
    """
    Gathers tensors from all processes and concatenates them.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Gather sizes of tensors
    local_size = torch.tensor([tensor.size(0)], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    # Calculate the maximum size
    max_size = max([size.item() for size in sizes])

    # Pad the tensors to have the same size
    padded_tensor = torch.zeros(max_size, *tensor.size()[1:], device=device)
    padded_tensor[:tensor.size(0)] = tensor

    # Gather all tensors
    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)

    # Remove padding and concatenate
    result = []
    for i, size in enumerate(sizes):
        result.append(gathered_tensors[i][:size.item()])
    return torch.cat(result, dim=0)

pro_name = ['sp', 'ssrd', 'strd', 't2m', 'tp', 'u10', 'v10', 'q', 'co2']
class Pipeline():
    def __init__(self, rank, args=None):
        self.rank = rank
        config = {
            'batch_size': 128,
            'epochs': 1000,
            'lr': 2e-4 * 16,
            'weight_decay': 1e-3,
            'seed': 0,
            'early_stop_round': 10,
            'latitude': 360,
            'longitude': 720,
            'save_model': True,
            'use_cuda': True,
            'input_dim': 0,
            'add_history_target': True,
            'kfold': 5,
            'kfold_shuffle': True,
        }
        self.config = config
        self.config.update(args.__dict__) 

        self.config['columns'] = ['year', 'month', 'latitude', 'longitude', self.config['target'],'sp', 'ssrd', 'strd', 't2m', 'tp', 'u10', 'v10', 'q', 'co2',]

        self.config['norm_columns'] = [self.config['target'], 'sp', 'ssrd', 'strd', 't2m', 'tp', 'u10', 'v10', 'q', 'co2']

        print(self.config)
        torch.manual_seed(config['seed'])
        self.device = torch.device(f'cuda:{rank}' if config['use_cuda'] else "cpu")
        
        self.normalizer = None
        self.models = []


    def normalize(self, data):
        if self.normalizer is None:
            self.normalizer = {}
            for col in self.config['norm_columns']:
                self.normalizer[col] = RobustScaler()
                self.normalizer[col].fit(data[col].values.reshape(-1, 1))

        for col in self.config['norm_columns']:
            data[col] = self.normalizer[col].transform(
                data[col].values.reshape(-1, 1)
            ).reshape(-1, )

        return data
    
    def preprocess_data(self, data, config, mode, start_year = 2008, end_year = 2019, target_upper = 700):
        """ 
            return:
                data:   np.array with shape (unique_latitude, unique_longitude, year+month, feature+label)
        """
        start_year = start_year - 1 if config['window'] == 13 else start_year
        data = data[(start_year <= data.year) & (data.year <= end_year)]
        print(data[config["target"]].mean())
        if np.abs(data[config["target"]].mean()) < 1e-6:
            data[config["target"]] = data[config["target"]] * 1e10
        print('normalize data...')
        data = self.normalize(data.copy())
        # TODO gc.collect()
        gc.collect()
        data[pro_name] = data[pro_name].fillna(0.0)

        for col in data.columns:
            if 'PFT' in col:
                data[col] = data[col].fillna(0.0)
        
        data = data.sort_values(by=['latitude', 'longitude', 'year', 'month'])
        feat_cnt = data.shape[-1]
        lat_cnt = data['latitude'].unique().shape[0]
        lon_cnt = data['longitude'].unique().shape[0]
        data[[config["target"]]] = np.clip(data[[config["target"]]], None, target_upper)
        data = data.values.astype(np.float32)
        data = data.reshape(lat_cnt, lon_cnt, -1, feat_cnt) # -1 conbines year and month
        print("data shape after preprocess: ", data.shape)
        return data

    def create_indexes(self, data, window = 13, lat_patch = 18, lon_patch = 18, lat_stride = None, lon_stride = None):
        """
        input:
            data: np.array with shape (unique_latitude, unique_longitude, year+month, feature+label)
            window: how many monthes will be involved in a sample
            lat_patch: length of latitude dim in a sample
            lon_patch: length of longitude dim in a sample
            lat_stride: gap of latitude sampling
            lon_stride: gap of longitude sampling
        output:
            indexes: 3 dimension np.array with each dimension means [latitude, longitude, year+month], which lists all possible index values
        """
        if lat_stride == None:
            lat_stride = lat_patch // 2
        if lon_stride == None:
            lon_stride = lon_patch // 2
        indexes = list(product(np.arange(0, data.shape[0] - lat_patch+1, lat_stride, dtype=int), 
                         np.arange(0, data.shape[1] - lon_patch+1, lon_stride, dtype=int), 
                         np.arange(0, data.shape[2] - window+1, dtype=int)))
        return indexes

    def build_dataloader_all_indexes(self, data, mode, start_year, end_year, shuffle=True , world_size=4):
        data = self.preprocess_data(data, self.config, mode, start_year, end_year)
        indexes = self.create_indexes(data, self.config['window'], self.config['patch'], self.config['patch'])
        dataset = OceanDataset_V2(data, indexes, self.config, mode)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=self.rank, drop_last=False, shuffle=shuffle) 
        dataloader = DataLoader(
            dataset, self.config['batch_size'],
            sampler=sampler,
            num_workers=0
        )
        return dataloader

    def run_epoch(self, model, data_loader, optimizer, epoch, scheduler=None):
        model.train()

        avg_sup_loss = 0.0
        avg_unsup_label_loss = 0.0

        with tqdm(data_loader, mininterval=3, disable=self.config['disable_tqdm']) as bar:
            for batch_id, batch in enumerate(bar):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                
                # weak aug
                data_w = augment(data, method='mask', intensity=0.1)
                # strong aug
                data_s = augment(data, method='mask', intensity=0.3)

                mask = 1 - torch.isnan(target).float()

                # supervised part
                out_w = model(data_w)
                sup_loss = mse_loss_with_nan(out_w, target, mask=mask)

                # unsupervised part
                out_s = model(data_s)
                unsup_loss = mse_loss_with_nan(out_s, out_w, mask=1-mask)

                loss = sup_loss + unsup_loss * 0.2

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                avg_sup_loss += sup_loss.item()
                avg_unsup_label_loss += unsup_loss.item()

                bar.set_postfix(
                    avg_label_loss=avg_sup_loss/(batch_id + 1),
                    avg_pseudo_label_loss=avg_unsup_label_loss/(batch_id + 1),
                )

    def eval_on(self, model, dataloader):
        model.eval()

        index = []
        predictions = []
        targets = []

        for batch_id, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                pred = model(data)
            index.append(data[..., -1, :4].data.cpu().numpy())
            predictions.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        index = np.concatenate(index, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        index = index.reshape(-1, index.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, target.shape[-1])

        predictions = self.normalizer[self.config["target"]].inverse_transform(predictions).reshape(-1, )
        targets = self.normalizer[self.config["target"]].inverse_transform(targets).reshape(-1, )
        result = pd.DataFrame({
            'year': index[:, 0],
            'month': index[:, 1],
            'lat': index[:, 2],
            'lon': index[:, 3],
            'predictions': predictions,
            'targets': targets
        })
        result = result.groupby(['year', 'month', 'lat', 'lon']).mean()

        eval_result = result.dropna()
        print(eval_result.shape)
        loss = np.mean(
            (eval_result['predictions'] - eval_result['targets']) ** 2
        )

        return loss


    def run(self, fold_idx=0):
        # initialize student model
        model_class = importlib.import_module(
            f'model.{self.config["model"]}'
        )
        model = model_class.Net(self.config).to(self.device)
        model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
        # load old check point if existed
        # check point would save to checkpointdir/exp_id/fold_idx epoch_idx.pth
        checkpoint_path = os.path.join(self.config['dump_dir'], str(self.config["exp_id"]), str(fold_idx))
        latest_epoch = 0
        finish =  False
        if os.path.exists(checkpoint_path):
            # check file
            for file in os.listdir(checkpoint_path):
                if 'finish' in file:
                    # finished, skip
                    finish = True
                    self.config['model_path'] = os.path.join(checkpoint_path, file)
                    break
                else:
                    cur_epoch = int(file.split('.')[0])
                    if cur_epoch > latest_epoch:
                        latest_epoch = cur_epoch
        if finish == True:
            return 
        
        if latest_epoch != 0:
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{latest_epoch}.pth')))

        
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        warmup_epochs = 5
        # Warm-up scheduler
        warmup_scheduler = WarmUpLR(optimizer, warmup_epochs=warmup_epochs, base_lr=0.01 * self.config['lr'], final_lr=self.config['lr'])

        # Cosine annealing scheduler to follow warm-up
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=100)



        self.best_val = np.inf
        self.early_stop_cnt = 0
        self.best_model = None

        patience = self.config['early_stop_round']

        print('start training...')

        for epoch in range(latest_epoch, self.config['epochs']):
            if epoch < warmup_epochs:
                scheduler = warmup_scheduler
            else:
                scheduler = cosine_scheduler
            self.run_epoch(model, self.train_loader, optimizer, epoch, scheduler=scheduler)
            val_metric = self.eval_on(model, self.valid_loader)
            loss_tensor = torch.tensor(val_metric).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # TODO can't use mean directly
            val_metric = loss_tensor.item() / dist.get_world_size()

            print(f'Epoch {epoch}, validation metric {val_metric}')

            if val_metric < self.best_val:
                self.best_val = val_metric
                self.best_model = copy.deepcopy(model)
                patience = self.config['early_stop_round']
            else:
                patience -= 1

            if patience == 0:
                break

            if epoch!=0 and epoch % 30 == 0:
                if self.config['dump_dir'] and self.rank == 0:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    shuffle = 'shuffle' if self.config['kfold_shuffle'] else 'unshuffle'
                    torch.save(self.best_model.state_dict(), os.path.join(
                        checkpoint_path, f'{epoch}.pth'))
                    print(f'dump model at {epoch} epoch')

        if self.config['dump_dir'] and self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(self.best_model.state_dict(), os.path.join(
                checkpoint_path, f'finish_{time.strftime("%Y-%m-%d_%H:%M", time.localtime())}_{self.best_val:.2f}.pth'))
            print('save model finished!')

    def test(self, is_evaluate=False, start_year=None, end_year=None):
        print('test start...')
        # load model
        if hasattr(self, 'best_model') and self.best_model:
            model = self.best_model
            print('load model from best model')
        else:
            assert ('model_path' in self.config.keys() ) and isinstance(
                self.config['model_path'], str), 'model_path should be provided'
            model_class = importlib.import_module(
                f'model.{self.config["model"]}'
            )
            print("Will load model from ", self.config['model_path'])
            model = model_class.Net(self.config).to(self.device)

            model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
           
            model.load_state_dict(torch.load(self.config['model_path']))

            self.best_model = model
        model.eval()
        assert type(model) == DDP, 'model should be DistributedDataParallel'

        test_loader = self.test_loader
        index = []
        predictions = []
        targets = []

        for batch_id, (data, target) in enumerate(test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                pred = model(data)

            # year, month, lat, lon
            index.append(data[..., -1, :4].data.cpu().numpy())
            predictions.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        index = np.concatenate(index, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        index = index.reshape(-1, index.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, target.shape[-1])
        print('transfer numpy array to torch tensor')
        predictions = self.normalizer[self.config["target"]].inverse_transform(predictions).reshape(-1, )
        targets = self.normalizer[self.config["target"]].inverse_transform(targets).reshape(-1, )

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world_size > 1:
            # Ensure tensors are on the same device
            index_tensor = torch.tensor(index).to(self.device)
            predictions_tensor = torch.tensor(predictions).to(self.device)
            targets_tensor = torch.tensor(targets).to(self.device)

            # Gather results from all processes
            gathered_index = gather_tensors(index_tensor, self.device)
            gathered_predictions = gather_tensors(predictions_tensor, self.device)
            gathered_targets = gather_tensors(targets_tensor, self.device)
            print('convert torch tensor to numpy array')
            # Convert to numpy arrays
            index = gathered_index.cpu().numpy()
            predictions = gathered_predictions.cpu().numpy()
            targets = gathered_targets.cpu().numpy()
        else:
            print('single process: skip distributed gather')

        result = pd.DataFrame({
            'year': index[:, 0],
            'month': index[:, 1],
            'lat': index[:, 2],
            'lon': index[:, 3],
            'predictions': predictions,
            'targets': targets
        })
        result = result.groupby(['year', 'month', 'lat', 'lon']).mean()
        if self.rank == 0:
            if start_year and end_year:
                result.to_csv(f'{self.config["result_dir"]}/{self.config["exp_id"]}_{start_year}-{end_year}.csv')
            else:
                result.to_csv(f'{self.config["result_dir"]}/{self.config["exp_id"]}.csv')
        
        if is_evaluate:
            eval_result = result.dropna()
            loss = np.mean(
                (eval_result['predictions'] - eval_result['targets']) ** 2
            )
            loss_tensor = torch.tensor(loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # TODO can't use mean directly
            loss = loss_tensor.item() / dist.get_world_size()
            print(f'test loss MSE: {loss}')
        
        print('test finished...')


    def plot_loss():
        pass

    def plot_prediction():
        pass

    def kfold_test(self, save_file=True):
        print('kfold test start...')
        
        total_result = pd.DataFrame()
        for i in range(self.config['kfold']):
            model = self.models[i]
            if type(model) != DDP:
                model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
            model.eval()

            index = []
            predictions = []
            targets = []

            for batch_id, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    pred = model(data)

                index.append(data[..., -1, :4].data.cpu().numpy())
                predictions.append(pred.data.cpu().numpy())
                targets.append(target.data.cpu().numpy())
            index = np.concatenate(index, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)

            index = index.reshape(-1, index.shape[-1])
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, target.shape[-1])

            predictions = self.normalizer[self.config["target"]].inverse_transform(
                predictions).reshape(-1, )
            targets = self.normalizer[self.config["target"]].inverse_transform(
                targets).reshape(-1, )

            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            if world_size > 1:
                index_tensor = torch.tensor(index).to(self.device)
                predictions_tensor = torch.tensor(predictions).to(self.device)
                targets_tensor = torch.tensor(targets).to(self.device)
                gathered_index = gather_tensors(index_tensor, self.device)
                gathered_predictions = gather_tensors(predictions_tensor, self.device)
                gathered_targets = gather_tensors(targets_tensor, self.device)

                index = gathered_index.cpu().numpy()
                predictions = gathered_predictions.cpu().numpy()
                targets = gathered_targets.cpu().numpy()
            else:
                print(f'single process: skip distributed gather for fold {i}')

            result = pd.DataFrame({
                'year': index[:, 0],
                'month': index[:, 1],
                'lat': index[:, 2],
                'lon': index[:, 3],
                'predictions': predictions,
                'targets': targets
            })
            result = result.groupby(['year', 'month', 'lat', 'lon']).mean()
            if save_file and self.rank == 0:
                result.to_csv(
                    f'{self.config["result_dir"]}/{self.config["exp_id"]}_{i}.csv')
            if i==0:
                total_result = result
            else:
                total_result = total_result + result

        print('save result')
        mean_result = total_result / self.config['kfold']
        if save_file and self.rank == 0:
            mean_result.to_csv(
                    f'{self.config["result_dir"]}/{self.config["exp_id"]}.csv')

        print('test finished...')
        return mean_result


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print('rank:', rank)
    torch.cuda.set_device(rank)
def end():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, help='window size', default=1)
    parser.add_argument('--patch', type=int, help='patch size', default=18)
    parser.add_argument('--model', type=str, help='model name', default='cnn')
    parser.add_argument('--percent', type=float, default=1.0, help='Percentage of samples with maximum non-missing rate')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--stage', type=str, default='train')
    parser.add_argument('--target', type=str, default='ORCHIDEE_S2_nbp')
    parser.add_argument('--exp_id', type=int, default=601)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data_name', type=str, default='data25_0616')
    parser.add_argument('--kfold_start_year', type=int, default=2000)
    parser.add_argument('--kfold_end_year', type=int, default=2023)
    parser.add_argument('--dump_dir', type=str, default='./checkpoint/')
    parser.add_argument('--result_dir', type=str, default='./result/')
    parser.add_argument('--preprocessed_data_dir', type=str, default='./data/')
    parser.add_argument('--pft_data_dir', type=str, default='./data/PFT/')

    args = parser.parse_args()
    mt = Pipeline(rank, args=args)  # supervised/semi_supervised
    data = get_data(mode=args.data_name, config=mt.config)

    mt.config['input_dim'] = data.shape[1] - 1
    print('input_dim', mt.config['input_dim'])

    if mt.config['stage'] == 'train':
        if mt.config['kfold'] > 1:

            kfold_start_year, kfold_end_year = args.kfold_start_year, args.kfold_end_year
            train_data = mt.preprocess_data(data, mt.config, mode='train', start_year=kfold_start_year, end_year=kfold_end_year)
            kf = KFold(n_splits=mt.config['kfold'], shuffle=mt.config['kfold_shuffle'], random_state=42)
            indexes = np.array(mt.create_indexes(train_data, mt.config['window'], mt.config['patch'], mt.config['patch']))
            print('index shape',indexes.shape)

            fold_idx = 0
            for train_idx, valid_idx in kf.split(indexes):
                print(train_idx.shape, valid_idx.shape)
                print('build train dataset')
                train_dataset = OceanDataset_V2(train_data, indexes[train_idx], mt.config, mode='train')  # Construct dataset
                sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=False, shuffle=True)
                mt.train_loader = DataLoader(
                    train_dataset, mt.config['batch_size'],
                    sampler=sampler, 
                    num_workers=0
                )

                print('build valid dataset')
                valid_dataset = OceanDataset_V2(train_data, indexes[valid_idx], mt.config, mode='valid')  # Construct dataset
                sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, drop_last=False, shuffle=True)
                mt.valid_loader = DataLoader(
                    valid_dataset, mt.config['batch_size'],
                    sampler=sampler, 
                    num_workers=0
                )
                mt.run(fold_idx=fold_idx)
                
                print(f'fold {fold_idx + 1} finished')
                fold_idx += 1
                del train_dataset
                del valid_dataset
                mt.train_loader = None
                mt.valid_loader = None
                gc.collect()

            del indexes
            gc.collect()
            

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

            mt.test_loader = mt.build_dataloader_all_indexes(data, 'valid', 2010, 2023, shuffle=False)
            mt.kfold_test()
        else:
            mt.train_loader = mt.build_dataloader_all_indexes(data, 'train', 2008, 2018, shuffle=True)
            mt.valid_loader = mt.build_dataloader_all_indexes(data, 'valid', 2019, 2020, shuffle=False)
            mt.run(fold_idx=0)

            mt.test_loader = mt.build_dataloader_all_indexes(data, 'valid', 2010, 2023, shuffle=False)
            mt.test()

    elif mt.config['stage'] == 'test':
        if mt.config['kfold'] > 1:
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
            test_start_year, test_end_year = 2020, 2023
            mt.test_loader = mt.build_dataloader_all_indexes(data, 'valid', test_start_year, test_end_year)

            mt.kfold_test()

        else:
            model_paths = glob.glob(mt.config['dump_dir'] + f'/*{mt.config["exp_id"]}/0/finish_*')
            print(model_paths)
            mt.config['model_path'] = model_paths[0]
            
            test_start_year, test_end_year = 2010, 2022
            test_start_year = test_start_year-1 if mt.config['add_history_target'] else test_start_year
            mt.test_loader = mt.build_dataloader_all_indexes(data, 'valid', test_start_year, test_end_year)
            mt.test(start_year=test_start_year, end_year=test_end_year)
    end()




from torch.utils.data import DistributedSampler
import torch.distributed as dist
if __name__ == '__main__':
    import os
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    main(rank, world_size)

    





