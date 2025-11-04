import torch
import torch.nn as nn
from src.layers.decomp import DECOMP
from src.layers.network import Network
from src.layers.revin import RevIN

class xPatch(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, patch_len, stride, padding_patch, revin, ma_type, alpha, beta):
        super(xPatch, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = enc_in

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # Normalization
        self.revin = revin
        self.revin_layer = RevIN(self.c_in, affine=True, subtract_last=False)

        # Moving Average
        self.ma_type = ma_type
        self.alpha = alpha       # smoothing factor for EMA (Exponential Moving Average)
        self.beta = beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, self.alpha, self.beta)
        self.net = Network(self.seq_len, self.pred_len, self.patch_len, self.stride, self.padding_patch)

    def forward(self, x):
        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)

        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x[:, :, -1]

if __name__ == '__main__':
    model = xPatch(
        seq_len = 50, pred_len = 50, enc_in = 6, patch_len = 10, stride = 5, padding_patch = 'end', revin = 1,
        ma_type = 'ema', alpha = 0.3, beta = 0.3
    ).to('cuda')

    x = torch.randn(64, 50, 6).to('cuda')
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

