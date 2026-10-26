import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


class Net(nn.Module):
    def __init__(self, config, hid_dim=64):
        super(Net, self).__init__()

        self.input_dim = (config['input_dim']+1) if config['add_history_target'] == True else config['input_dim']
        self.hid_dim = hid_dim
        self.lat = config['latitude']
        self.lon = config['longitude']
        self._build_em()
        self._build_enc()

    def _build_em(self):
        self.inputs_em = nn.Linear(self.input_dim - 4, self.hid_dim)
        self.year_em = nn.Embedding(100, self.hid_dim)
        self.mon_em = nn.Embedding(12 + 1, self.hid_dim)
        self.lat_em = nn.Embedding(self.lat, self.hid_dim)
        self.lon_em = nn.Embedding(self.lon, self.hid_dim)

    def _build_enc(self):
        self.cnn_blocks_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, padding='same'),
                nn.GroupNorm(num_groups=8, num_channels=self.hid_dim),
                nn.GELU(),
                nn.Dropout(p=0.25)
            ) for l in range(5)
        ])

        self.rnn = nn.GRU(self.hid_dim, self.hid_dim)

        self.out = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.GELU(),
            nn.Linear(self.hid_dim, 1)
        )

    def embed_time(self, x):
        month_em = self.mon_em(x.select(-1, 1).long())
        time_em = month_em
        return time_em

    def embed_loc(self, x):
        lat_em = self.lat_em(x.select(-1, 2).long())
        lon_em = self.lon_em(x.select(-1, 3).long())
        loc_em = lat_em + lon_em
        return loc_em

    def embed_inputs(self, x):
        inputs_em = self.inputs_em(x[..., 4:])
        return inputs_em

    def forward(self, x):
        B, H, W, L, _ = x.size()

        time_em = self.embed_time(x)
        loc_em = self.embed_loc(x)
        inputs_em = self.embed_inputs(x)
        x = time_em + loc_em + inputs_em

        xr = einops.rearrange(x, 'b h w l f -> l (b h w) f')
        xr, _ = self.rnn(xr)
        xr = einops.rearrange(xr, 'l (b h w) f -> b h w l f', b=B, h=H, w=W)

        x = x + xr

        # conv net
        xc = einops.rearrange(x, 'b h w l f -> (b l) f h w')
        for conv_layer in self.cnn_blocks_1:
            xc = xc + conv_layer(xc)
        xc = einops.rearrange(xc, '(b l) f h w -> b h w l f', b=B)

        x = self.out(x).select(-2, -1)

        return x
