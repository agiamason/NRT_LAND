import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import einops  
  
class TransformerBlock(nn.Module):  
    def __init__(self, hid_dim, n_heads, dropout=0.1):  
        super(TransformerBlock, self).__init__()  
        self.attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads)  
        self.norm1 = nn.LayerNorm(hid_dim)  
        self.norm2 = nn.LayerNorm(hid_dim)  
        self.ff = nn.Sequential(  
            nn.Linear(hid_dim, hid_dim * 2),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(hid_dim * 2, hid_dim),  
            nn.Dropout(dropout)  
        )  
  
    def forward(self, x):  
        attn_out, _ = self.attention(x, x, x)  
        x = self.norm1(attn_out + x)  
        ff_out = self.ff(x)  
        x = self.norm2(ff_out + x)  
        return x  
  
class Net(nn.Module):  
    def __init__(self, config, hid_dim=64, n_heads=4, n_layers=4):  
        super(Net, self).__init__()  
        self.input_dim = (config['input_dim'] + 1) if config['add_history_target'] else config['input_dim']  
        self.window = config['window']  
        self.hid_dim = hid_dim  
        self.lat = config['latitude']  
        self.lon = config['longitude']  
          
        self._build_em()  
        self._build_enc(n_heads, n_layers)  
  
    def _build_em(self):  
        self.inputs_em = nn.Linear(self.input_dim - 4, self.hid_dim)  
        self.year_em = nn.Embedding(100, self.hid_dim)  
        self.mon_em = nn.Embedding(12 + 1, self.hid_dim)  
        self.lat_em = nn.Embedding(self.lat, self.hid_dim)  
        self.lon_em = nn.Embedding(self.lon, self.hid_dim)  
  
    def _build_enc(self, n_heads, n_layers):  
        self.transformer_blocks = nn.ModuleList(  
            [TransformerBlock(self.hid_dim, n_heads) for _ in range(n_layers)]  
        )  
        self.out = nn.Sequential(  
            nn.Linear(self.hid_dim, self.hid_dim),  
            nn.GELU(),  
            nn.Linear(self.hid_dim, 1)  
        )  
  
    def embed_inputs(self, x):  
        month_em = self.mon_em(x.select(-1, 1).long())  
        lat_em = self.lat_em(x.select(-1, 2).long())  
        lon_em = self.lon_em(x.select(-1, 3).long())  
        loc_em = lat_em + lon_em  
        inputs_em = self.inputs_em(x[..., 4:])  
        return inputs_em + month_em + loc_em  
    
    def forward(self, x):  
        B, H, W, L, _ = x.size()  
        x = self.embed_inputs(x)  
        
        x = einops.rearrange(x, 'b h w l f -> (h w l) b f') 

        for layer in self.transformer_blocks:  
            x = layer(x)  
        x = einops.rearrange(x, '(h w l) b f -> b h w l f', h=H, w=W)  
        x = x.mean(dim=-2)  
        x = self.out(x)  # (batch_size, output_dim)  
        return x  
