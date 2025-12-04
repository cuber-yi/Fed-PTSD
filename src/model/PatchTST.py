import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, patch_len=16, stride=8, d_model=128, n_heads=4, num_layers=3,
                 dropout=0.1):
        super(PatchTST, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.enc_in = enc_in

        # Patching calculations
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_num += 1

        # Backbone (Channel Independent: treat all channels as batch dim)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.patch_num, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=True)

        # [修改]: 添加 enable_nested_tensor=False 以消除警告
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                         enable_nested_tensor=False)

        # Head
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(self.patch_num * d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        B, S, C = x.shape

        # Channel Independence: [B, S, C] -> [B*C, S, 1]
        x = x.permute(0, 2, 1).reshape(B * C, S, 1)  # [B*C, S, 1]

        # Patching
        x = x.squeeze(-1)  # [B*C, S]
        # 简单Pad策略
        if self.stride > 0:
            x = self.padding_patch_layer(x.unsqueeze(1)).squeeze(1)

        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B*C, Patch_Num, Patch_Len]

        # Embedding
        x = self.patch_embedding(x)  # [B*C, Num, d_model]
        x = x + self.positional_encoding[:, :x.shape[1], :]

        # Transformer
        x = self.transformer_encoder(x)

        # Head
        x = self.flatten(x)
        x = self.head(x)  # [B*C, Pred]

        # Reshape back
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, Pred, C]

        # 如果是单变量任务目标，做简单聚合
        return x.mean(dim=2)
