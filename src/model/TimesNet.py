import torch
import torch.nn as nn
import torch.fft


class TimesNet(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, d_model=64, num_layers=2, k=3):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = k

        self.embedding = nn.Linear(enc_in, d_model)
        self.layers = nn.ModuleList([TimesBlock(d_model, k) for _ in range(num_layers)])

        # Prediction
        self.projection = nn.Linear(d_model, enc_in)
        self.predict_linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        x = self.embedding(x)  # [B, S, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, S]

        for layer in self.layers:
            x = layer(x)

        x = self.predict_linear(x)  # [B, d_model, Pred]
        x = x.permute(0, 2, 1)  # [B, Pred, d_model]
        x = self.projection(x)  # [B, Pred, Channel]

        # SOH 预测聚合
        if x.shape[2] == 1:
            return x.squeeze(-1)
        return x.mean(dim=2)


class TimesBlock(nn.Module):
    def __init__(self, d_model, k):
        super(TimesBlock, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_model),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, d_model, S]
        B, D, S = x.size()

        # FFT to find top-k periods
        x_fft = torch.fft.rfft(x, dim=-1)
        freq_list = abs(x_fft).mean(0).mean(0)
        freq_list[0] = 0
        _, top_list = torch.topk(freq_list, self.k)
        top_list = top_list.detach().cpu().numpy()

        res = x
        for i in range(self.k):
            period = S // top_list[i]
            # Padding
            if S % period != 0:
                length = ((S // period) + 1) * period
                padding = torch.zeros([B, D, (length - S)]).to(x.device)
                out = torch.cat([x, padding], dim=-1)
            else:
                length = S
                out = x

            # Reshape to 2D
            # [修正]: 移除 permute，保持 (B, D, H, W) 格式以适配 Conv2d
            out = out.reshape(B, D, length // period, period).contiguous()
            out = self.conv(out)
            out = out.reshape(B, D, -1)

            x = x + out[..., :S]  # Residual

        return x


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_Block_V1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x4 = self.conv7(x)
        return x1 + x2 + x3 + x4
