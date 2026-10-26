import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class Net(nn.Module):
    def __init__(self, input_dim, hid_dim=64, config=None):

        super(Net, self).__init__()

        self.input_dim = input_dim
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
        self.enc = Encoder(self.hid_dim, 16, 4)
        self.dec = Decoder(16, self.hid_dim, 4)
        self.hid = Mid_Xnet(6*16, 256, 4, [3, 5, 7, 11], 8)

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
        x = self.embed_inputs(x)

        x = einops.rearrange(x, 'b h w l f -> (b l) f h w')
        x = F.pad(x, (0, 2, 0, 2))

        embed, skip = self.enc(x)
        embed = einops.rearrange(embed, '(b l) f h w -> b l f h w', b=B)

        hid = self.hid(embed)
        hid = einops.rearrange(hid, 'b l f h w -> (b l) f h w')

        x = self.dec(hid, skip)

        x = x[..., :H, :W]

        x = einops.rearrange(x, '(b l) f h w -> b h w l f', b=B, h=H, w=W)

        x = self.out(x)

        return x.select(-2, -1)
