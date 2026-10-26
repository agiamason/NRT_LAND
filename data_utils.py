import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pyarrow.feather as feather

class OceanDataset(Dataset):
    def __init__(self, inputs, outputs, config, mode):
        inputs, outputs = self.slice_patch(inputs, outputs, patch_size=config['patch'])
        if mode == 'train':
            inputs, outputs = self.select_top_rows(inputs, outputs, config['percent'])
        inputs = inputs.astype(np.float32)
        outputs = outputs.astype(np.float32)
        self.data = torch.from_numpy(inputs).float()
        self.target = torch.from_numpy(outputs).float()
        print('self.data.dtype',self.data.dtype)

    def select_top_rows(self, inputs, outputs, percent):
        nan_rate = self.get_nan_rate(outputs)
        top_row_idx = np.argsort(nan_rate)
        top_row_idx = top_row_idx[:int(top_row_idx.shape[0] * percent)]
        inputs, outputs = inputs[top_row_idx], outputs[top_row_idx]
        return inputs, outputs

    def get_nan_rate(self, outputs):
        outputs = outputs.reshape(outputs.shape[0], -1)
        nan_rate = np.isnan(outputs).astype('float')
        nan_rate = nan_rate.sum(axis=-1) / outputs.shape[-1]
        return nan_rate

    def slice_patch(self, inputs, outputs, patch_size=18):
        B, H, W, L, _ = inputs.shape
        patch_inputs = []
        patch_outputs = []

        stride = patch_size // 2

        for i in range(0, H - patch_size+1, stride):
            for j in range(0, W - patch_size+1, stride):
                _inputs = inputs[:, i: i+patch_size, j: j+patch_size]
                _outputs = outputs[:, i: i+patch_size, j: j+patch_size]
                patch_inputs.append(_inputs)
                patch_outputs.append(_outputs)

        patch_inputs = np.concatenate(patch_inputs, axis=0)
        patch_outputs = np.concatenate(patch_outputs, axis=0)

        return patch_inputs, patch_outputs

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def dataloader(data, config, shuffle, n_jobs=0, mode='train', train_shuffle=True, valid_shuffle=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    data = data.sort_values(by=['latitude', 'longitude', 'year', 'month'])
    
    inputs = data.drop([config["target"]], axis=1).values
    outputs = data[[config["target"]]].values
    outputs = np.where(outputs>700, 700, outputs)


    inputs = inputs.astype(np.float32)
    outputs = outputs.astype(np.float32)
    print('data.dtype',inputs.dtype, outputs.dtype)

    lat_cnt = data['latitude'].unique().shape[0]
    lon_cnt = data['longitude'].unique().shape[0]
    feat_cnt = data.shape[-1]

    inputs = inputs.reshape(lat_cnt, lon_cnt, -1, feat_cnt - 1)
    outputs = outputs.reshape(lat_cnt, lon_cnt, -1, 1) 

    inputs = np.array([inputs[..., i - config['window']: i, :] for i in range(config['window'], inputs.shape[2] + 1)])
    
    
    if config['add_history_target'] == True:
        #inputs concat history target
        input_add = np.array([outputs[..., i - config['window']: i, :] for i in range(config['window'], outputs.shape[2] + 1)])
        input_add[:,:,:,(input_add.shape[3] - 1),:] = 0
        # TODO add mask 
        input_add = np.nan_to_num(input_add, nan=0)
        inputs = np.concatenate((inputs, input_add), axis=4)
    outputs = np.array([outputs[..., i - 1, :] for i in range(config['window'], outputs.shape[2] + 1)])
    print(inputs.shape, outputs.shape)
    if mode == 'train':
        batch_size = config['batch_size']
    else:
        batch_size = config['batch_size'] * 10

    return inputs, outputs


def get_data(mode, config):
    print('target:', config['target'])
    preprocessed_path = os.path.join(config['preprocessed_data_dir'], '{}.feather'.format(mode))

    print('read preprocessed data')
    data = feather.read_feather(preprocessed_path, memory_map=True)
    data = data[data['year']>=2000][config['columns']]
    data = optimize_floats(data)

    pft = feather.read_feather(os.path.join(config['pft_data_dir'], f"{config['target']}.feather"), memory_map=True)
    pft = optimize_floats(pft)
    assert pft.shape[0] == data.shape[0], 'PFT data length should equal to data length'
    
    for col in pft.columns:
        if 'PFT' in col:
            data[col] = pft[col].values
    print(data.columns)
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
