import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, hidden_size=128, num_layers=2):
        """
        enc_in: 输入特征维度
        pred_len: 预测长度
        seq_len: 输入序列长度 (LSTM模型中未使用，但为保持接口一致性而保留)
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        """
        super(LSTM, self).__init__()
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: [Batch, seq_len, enc_in]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_time_step_out = lstm_out[:, -1, :]

        out = self.head(last_time_step_out)

        return out
