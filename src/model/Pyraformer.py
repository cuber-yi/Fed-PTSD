import torch
import torch.nn as nn


class Pyraformer(nn.Module):
    """
    Pyraformer: Pyramidal Attention based Transformer for Time Series (Simplified)
    """

    def __init__(self, enc_in, pred_len, seq_len, d_model=64, n_heads=4, num_layers=2, window_size=[2, 2]):
        super(Pyraformer, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len

        # Embedding
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.down_convs = nn.ModuleList()
        current_seq_len = seq_len
        for w in window_size:
            self.down_convs.append(nn.Conv1d(d_model, d_model, kernel_size=w, stride=w))
            current_seq_len //= w

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Projection
        # 由于金字塔结构会改变长度，我们这里使用Flatten+Linear作为简单的解码头
        # 聚合多尺度特征
        total_len = seq_len
        curr = seq_len
        for w in window_size:
            curr //= w
            total_len += curr

        self.projection = nn.Linear(total_len * d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        batch_size = x.shape[0]

        # Embedding
        x_enc = self.enc_embedding(x)  # [Batch, Seq, d_model]

        # Multi-scale processing
        scales = [x_enc]
        curr_x = x_enc.permute(0, 2, 1)  # [B, C, L]

        for conv in self.down_convs:
            curr_x = conv(curr_x)
            scales.append(curr_x.permute(0, 2, 1))

        # Concatenate all scales (Pyramidal Token Sequence)
        # [Batch, Seq + Seq/2 + Seq/4 ..., d_model]
        multi_scale_x = torch.cat(scales, dim=1)

        # Encoder
        enc_out = self.transformer_encoder(multi_scale_x)

        # Decoder / Prediction
        out = enc_out.reshape(batch_size, -1)
        out = self.projection(out)  # [Batch, Pred]

        return out
