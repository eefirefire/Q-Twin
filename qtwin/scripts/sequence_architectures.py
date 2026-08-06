"""
Week 4 Task 4: additional sequence-model architectures beyond the Week 3
LSTM baseline (model_trainer.py's SequenceLSTM), for lstm_tcn_comparison.py.

Two variants, both aimed specifically at the Week 3 LSTM's known weak
point (DIVERGENT_REPLICATES: 4/7 misclassified as SUCCESS) rather than a
generic "try more architectures":

- LSTMAttention: same LSTM backbone, but instead of only using the final
  hidden state (which forces the whole 45s curve through one summary
  vector), an additive-attention layer lets the model learn to weight
  whichever time region actually distinguishes a divergent curve --
  useful because a corrupted replicate's flat-then-spike pattern can
  appear anywhere in the window, not necessarily at the end.
- TCN: dilated causal 1D convolutions. Unlike an LSTM's single sequential
  pass, a TCN's receptive field is built from local windows at multiple
  dilations simultaneously, which can be better suited to detecting a
  LOCAL pattern shift (the flat-then-spike divergence signature) rather
  than needing to carry it through a long recurrent state.
"""

from torch import nn
import torch

from gatekeeper_model import LABELS


class LSTMAttention(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, len(LABELS))

    def forward(self, x):
        # x: (batch, seq_len, 1)
        outputs, _ = self.lstm(x)  # (batch, seq_len, hidden)
        scores = self.attn_score(outputs).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (outputs * weights).sum(dim=1)  # (batch, hidden)
        out = self.dropout(context)
        return self.fc(out)


class _CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # causal: drop the right-side padding overhang
        return self.dropout(self.relu(out))


class TCN(nn.Module):
    def __init__(self, channels: int = 16, n_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(_CausalConvBlock(in_ch, channels, kernel_size, dilation))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, len(LABELS))

    def forward(self, x):
        # x: (batch, seq_len, 1) -> conv1d wants (batch, channels, seq_len)
        h = self.net(x.transpose(1, 2))  # (batch, channels, seq_len)
        pooled = h.mean(dim=2)  # global average pool over time
        return self.fc(pooled)
