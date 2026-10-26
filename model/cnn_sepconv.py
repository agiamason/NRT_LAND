import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


class Net(nn.Module):
    def __init__(self, config, hid_dim=64):
        super(Net, self).__init__()

        self.input_dim = config['input_dim']
        self.window = config['window']
        self.hid_dim = hid_dim
        self.lat = config['latitude']
        self.lon = config['longitude']
        self.env_dim = config['env_dim']
            
        self._build_em()
        self._build_enc()

    def _build_em(self):
        self.inputs_em = nn.Linear(self.input_dim - 4 - self.env_dim, self.hid_dim)
        self.env_em = nn.Linear(self.env_dim, self.hid_dim)
        self.year_em = nn.Embedding(100, self.hid_dim)
        self.mon_em = nn.Embedding(12 + 1, self.hid_dim)
        self.lat_em = nn.Embedding(self.lat, self.hid_dim)
        self.lon_em = nn.Embedding(self.lon, self.hid_dim)

    def _build_enc(self):
        self.input_conv = nn.Conv2d(self.hid_dim * self.window, self.hid_dim, kernel_size=1)

        self.cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, padding='same'),
                nn.GroupNorm(num_groups=8, num_channels=self.hid_dim),
                nn.GELU(),
                nn.Dropout(p=0.25)
            ) for l in range(5)
        ])

        self.out = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.GELU(),
            nn.Linear(self.hid_dim, 1)
        )

    def embed_inputs(self, x):
        month_em = self.mon_em(x.select(-1, 1).long())
        time_em = month_em

        lat_em = self.lat_em(x.select(-1, 2).long())
        lon_em = self.lon_em(x.select(-1, 3).long())
        loc_em = lat_em + lon_em

        inputs_em = self.inputs_em(x[..., 4:])

        return inputs_em + time_em + loc_em

    def forward(self, x):
        B, H, W, L, _ = x.size()

        if self.env_dim != 0:
            first_x =  x[..., :self.input_dim-self.env_dim]
            last_x =  x[..., -self.env_dim:]
            
            first_x = self.embed_inputs(first_x)
            # flatten win size to feature
            first_x = einops.rearrange(first_x, ' b h w (p l) f -> b h w p (l f)', p=1)
            first_x = einops.rearrange(first_x, 'b h w l f -> (b l) f h w')
            first_x = self.input_conv(first_x)
            
            
            # flatten win size to feature
            last_x = self.env_em(last_x)
            last_x = einops.rearrange(last_x, ' b h w (p l) f -> b h w p (l f)', p=1)
            last_x = einops.rearrange(last_x, 'b h w l f -> (b l) f h w')
            last_x = self.input_conv(last_x)
            
            x= first_x + last_x
            
        else:
            x = self.embed_inputs(x)

            # flatten win size to feature
            x = einops.rearrange(x, ' b h w (p l) f -> b h w p (l f)', p=1)

            x = einops.rearrange(x, 'b h w l f -> (b l) f h w')
            x = self.input_conv(x)

        for layer in self.cnn_blocks:
            layer_X = layer(x)
            x = x + layer_X
            
        x = einops.rearrange(x, '(b l) f h w -> b h w l f', b=B, h=H, w=W)

        x = self.out(x).select(-2, -1)

        return x
