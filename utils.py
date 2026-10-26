import os
import numpy as np
import pandas as pd
import netCDF4 as nc

import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import einops
from torch.optim.lr_scheduler import StepLR

def ramp_up(min_v, max_v, cur_t, MAX_T):
    cur_t = min(cur_t, MAX_T)
    return (max_v - min_v) / MAX_T * cur_t + min_v


def mse_loss_with_nan(x, y, mask):
    y = torch.nan_to_num(y)
    loss = F.mse_loss(x, y, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + 1e-3)
    return loss


def likelihood_with_mask(d, y, mask):
    y = torch.nan_to_num(y)
    p = -d.log_prob(y)
    p = (p * mask).sum() / (mask.sum() + 1e-3)
    return p


def kl_div_with_mask(p, q, mask):
    kl_div = kl_divergence(p, q)
    kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-3)
    return kl_div


def augment(x, method='mask', intensity=0.1):
    def _mask(x):
        mask = torch.rand_like(x)
        mask = (mask < 1 - intensity).float()
        return x * mask

    def _shuffle(x):
        index = torch.randperm(x.size(-1)).to(x.device)
        perm_x = torch.index_select(x, -1, index)

        return x * (1 - intensity) + perm_x * intensity

    cat = x[..., :4]
    num = x[..., 4:]

    if method == 'mask':
        num = _mask(num)
    if method == 'shuffle':
        num = _shuffle(num)

    x = torch.cat([cat, num], dim=-1)
    return x

def quantile_aug(x, quantile_num=10):
    B, H, W, L, _ = x.size()
    x = einops.rearrange(x, ' b h w l f -> (b l f) h w',  b=B, h=H, w=W, l=L)
    x= x.reshape(-1, H*W)
    
    quantile = torch.quantile(x, torch.tensor([i*(1/(quantile_num+1)) for i in range(quantile_num+2)]).to(x.device), dim=1, keepdim=True)
    idx = (x>=quantile[:quantile_num+1]).sum(dim=0)
    quantile = quantile.permute(1,0,2)
    quantile_new = (quantile[:,:quantile_num+1] + quantile[:,1:]) / 2
    x_new = torch.gather(quantile_new.squeeze(), 1, idx-1)
    x_new = einops.rearrange(x_new, ' (b l f) (h w)  -> b h w l f',  b=B, h=H, w=W, l=L)
    return x_new


def nc2csv(start_year, end_year, obs_path, pro_dir, target_path):
    # initiate dataframe with year*month*latitude*longitude
    latitude_len, longitude_len = 180, 360
    df = pd.MultiIndex.from_product([[year for year in range(start_year, end_year+1)],
                                     [month+1 for month in range(12)],
                                     [latitude for latitude in range(latitude_len)],
                                     [longitude for longitude in range(longitude_len)]],
                                    names=['year', 'month', 'latitude', 'longitude']).to_frame(index=False)

    # read observation data
    obs_data = nc.Dataset(obs_path)
    df['socat'] = obs_data.variables['observation data'][:].flatten()

    # read pro data
    for i in os.listdir(pro_dir):
        pro_data = nc.Dataset(os.path.join(pro_dir, i))
        key = i.split('.')[0]
        df[key] = pro_data.variables[key][:].flatten()

    df.to_csv(target_path)
    return df


def transfer_data():
    setting = {
        'train': (1959, 2013),
        'valid': (2014, 2015),
        'test': (2016, 2017)
    }
    for mode in setting.keys():
        start_year, end_year = setting[mode]
        obs_path = '../data/origin_split_data/obs_data_{}/obs.nc'.format(mode)
        pro_dir = '../data/origin_split_data/pro_data_{}'.format(mode)
        target_path = '../data/split_data/{}.csv'.format(mode)

        nc2csv(start_year, end_year, obs_path, pro_dir, target_path)


    

class StepLRWithMinLRScheduler:
    def __init__(self, optimizer, step_size, gamma, min_lr):
        self.step_lr = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.min_lr = min_lr
        self.optimizer = optimizer

    def step(self):
        self.step_lr.step()
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
        

from torch.optim.lr_scheduler import CosineAnnealingLR

class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.final_lr - self.base_lr) / self.warmup_epochs
            return [self.base_lr + warmup_factor * self.last_epoch for _ in self.optimizer.param_groups]
        else:
            return [self.final_lr for _ in self.optimizer.param_groups]
