import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pyarrow.feather as feather


class OceanDataset_V2(Dataset):

    def __init__(self, data, indexes, config, mode):
        self.config = config
        self.lat_patch = config['patch']
        self.lon_patch = config['patch']
        self.window_size = config['window']
        indexes = np.array(indexes)
        if mode == 'train':
            indexes = self.select_top_rows(data, indexes, config['percent'])
        data[..., :-1][np.isnan(data[..., :-1])] = 0
        self.data = torch.from_numpy(data.astype(np.float32))
        self.indexes = torch.from_numpy(indexes)
        print(f'{mode} dataset shape: {self.data.shape}, indexes shape: {self.indexes.shape}')

    def select_top_rows(self, data, indexes, percent):
        if percent == 1:
            return indexes
        nan_rate = self.get_nan_rate(data, indexes)
        top_row_idx = np.argsort(nan_rate)
        top_row_idx = top_row_idx[:int(top_row_idx.shape[0] * percent)]
        indexes = indexes[top_row_idx]
        return indexes

    def get_nan_rate(self, data, indexes):
        labels = [self.get_label(data, d3_index) for d3_index in indexes]
        labels = np.stack(labels, axis = 0).reshape(len(labels), -1)
        nan_rate = np.isnan(labels).astype('float')
        nan_rate = nan_rate.sum(axis=-1) / labels.shape[-1]
        return nan_rate

    def get_label(self, data, d3_index):
        latitude_start, longitude_start, month_start = d3_index
        label = data[latitude_start:latitude_start + self.lat_patch, 
                        longitude_start:longitude_start + self.lon_patch, 
                        month_start + self.window_size - 1, -1]
        return label

    def __getitem__(self, index):
                
        latitude_start, longitude_start, month_start = self.indexes[index]

        
        data = self.data[latitude_start:latitude_start + self.lat_patch, 
                        longitude_start:longitude_start + self.lon_patch, 
                        month_start:month_start + self.window_size, ...].clone()
    
        if self.config['add_history_target']:
            data[:, :, -1, -1] = 0
        else:
            data = data[..., :-1]
        data = torch.nan_to_num(data, nan=0.0)
        label = self.get_label(self.data, self.indexes[index]).unsqueeze(-1)
        torch.isnan(label)
        return data, label
    

    def __len__(self):
        return len(self.indexes)


def get_data(mode, config):
    print('target:', config['target'])
    preprocessed_path = os.path.join(config['preprocessed_data_dir'], '{}.feather'.format(mode))

    print('read preprocessed data', preprocessed_path)
    data = feather.read_feather(preprocessed_path, memory_map=True)
    data = data[data['year']>=2000][config['columns']]
    data = optimize_floats(data)

    pft_path = os.path.join(config['pft_data_dir'], f"{config['target']}.feather")
    if os.path.exists(pft_path):
        pft = feather.read_feather(pft_path, memory_map=True)
        pft = optimize_floats(pft)
        assert pft.shape[0] == data.shape[0], 'PFT data length should equal to data length'
        
        for col in pft.columns:
            if 'PFT' in col:
                data[col] = pft[col].values
    target_col = data.pop(config['target'])
    data[config['target']] = target_col
    print(f"data.columns: {data.columns}")

    return data

def optimize_floats(df):
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype('float32')
    return df


def add_prefix(data, prefix_data, L):
    prefix_data = prefix_data.sort_values(by=['latitude', 'longitude', 'year', 'month'])
    prefix_data = prefix_data.groupby(['latitude', 'longitude'], group_keys=False).apply(
        lambda x: x.iloc[-L:]
    ).reset_index()
    data = pd.concat([prefix_data, data], axis=0)
    return data
