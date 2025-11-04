import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, hidden_size=128, num_layers=2):

        super(RNN, self).__init__()
        self.pred_len = pred_len

        self.rnn = nn.RNN(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_time_step_out = rnn_out[:, -1, :]
        out = self.head(last_time_step_out)

        return out
