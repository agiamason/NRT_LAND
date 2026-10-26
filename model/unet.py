import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import einops

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size =3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size =3, padding=1)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs=(13,32,64,128,256,512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2, ceil_mode=False)

    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs



class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Net(nn.Module):
    def __init__(self, config, hid_dim=32,  enc_chs=(32,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), 
                    num_class=1, retain_dim=True):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        
        self.input_dim = self.input_dim = (config['input_dim']+1) if config['add_history_target'] == True else config['input_dim']
        self.hid_dim = hid_dim
        self.lat = config['latitude']
        self.lon = config['longitude']
        self.out_sz = (config['patch'], config['patch'])
        self._build_em()

    def _build_em(self):
        self.inputs_em = nn.Linear(self.input_dim - 4, self.hid_dim)
        self.year_em = nn.Embedding(100, self.hid_dim)
        self.mon_em = nn.Embedding(12 + 1, self.hid_dim)
        self.lat_em = nn.Embedding(self.lat, self.hid_dim)
        self.lon_em = nn.Embedding(self.lon, self.hid_dim)

    def embed_inputs(self, x):
        month_em = self.mon_em(x.select(-1, 1).long())
        time_em = month_em

        lat_em = self.lat_em(x.select(-1, 2).long())
        lon_em = self.lon_em(x.select(-1, 3).long())
        loc_em = lat_em + lon_em

        inputs_em = self.inputs_em(x[:, :, :, :, 4:])

        return inputs_em + time_em + loc_em

    def forward(self, x):
        x = self.embed_inputs(x)
        B, H, W, L, _ = x.size()
        x = einops.rearrange(x, 'b h w l f -> l b f h w', b=B, h=H, w=W, l=L)

        x = x.squeeze(0)

        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        out = out.unsqueeze(0)
        out = einops.rearrange(out, 'l b f h w -> b h w l f', b=B, h=H, w=W)
        out = out.squeeze(-1)
         
        return  out
    
