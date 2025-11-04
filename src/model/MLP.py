import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, hidden_size=256):
        """
        enc_in: 输入特征维度
        pred_len: 预测长度
        seq_len: 输入序列长度
        hidden_size: 隐藏层大小
        """
        super(MLP, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(seq_len * enc_in, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: [Batch, seq_len, enc_in]
        x_flat = self.flatten(x)
        out = self.fc1(x_flat)
        out = self.relu(out)
        out = self.fc2(out)

        return out
