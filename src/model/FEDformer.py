import torch
import torch.nn as nn
import torch.nn.functional as F


class FEDformer(nn.Module):
    """
    FEDformer: Frequency Enhanced Decomposed Transformer (Fourier Version)
    """

    def __init__(self, enc_in, pred_len, seq_len, d_model=64, n_heads=4, num_layers=2, modes=32):
        super(FEDformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Decomp
        self.decomp = SeriesDecomp(kernel_size=25)

        # Embedding
        self.enc_embedding = nn.Linear(enc_in, d_model)

        # Fourier Blocks
        self.layers = nn.ModuleList([
            FourierBlock(d_model, n_heads, modes=min(modes, seq_len // 2))
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)
        self.channel_map = nn.Linear(1, enc_in)  # 简单的通道映射

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        seasonal_init, trend_init = self.decomp(x)

        # Encoder
        enc_out = self.enc_embedding(seasonal_init)
        for layer in self.layers:
            enc_out = layer(enc_out)

        enc_out = self.norm(enc_out)

        # Project Time dim: [Batch, Seq, d_model] -> [Batch, Seq, Pred] -> [Batch, Pred, d_model]
        # 这里简化：直接 Flatten 映射 或者 只取最后部分
        # 为了高效，我们使用 Linear 映射时间维度
        enc_out = enc_out.permute(0, 2, 1)  # [B, d_model, Seq]
        enc_out = F.adaptive_avg_pool1d(enc_out, 1).squeeze(-1)  # [B, d_model]

        dec_out = self.projection(enc_out)  # [B, Pred]

        # 输出 [Batch, Pred] 适配 SOH 预测
        return dec_out


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        moving_mean = self.moving_avg(x_t)
        # Handle padding issue if necessary, here simplistic
        if moving_mean.shape[2] != x.shape[1]:
            moving_mean = F.interpolate(moving_mean, size=x.shape[1], mode='linear')
        moving_mean = moving_mean.permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean


class FourierBlock(nn.Module):
    def __init__(self, d_model, n_heads, modes):
        super(FourierBlock, self).__init__()
        self.d_model = d_model
        self.modes = modes
        # Frequency domain weights
        self.index_q = list(range(0, modes))
        self.scale = (1 / (d_model * d_model))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, d_model // n_heads, d_model // n_heads, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: [Batch, Seq, d_model]
        B, L, D = x.shape
        x_ft = torch.fft.rfft(x, dim=1)

        # Perform selection
        x_ft_sub = x_ft[:, :self.modes, :]

        # Simplified Frequency processing (No Cross-Attention for simple forecasting baseline)
        # Just modulation
        out_ft = x_ft.clone()
        # 这里仅做简单滤波示例，完整版需要复数矩阵乘法
        out_ft[:, :self.modes, :] = x_ft[:, :self.modes, :] * 0.5

        x = torch.fft.irfft(out_ft, n=L, dim=1)
        return x
