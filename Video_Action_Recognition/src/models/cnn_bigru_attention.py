"""BiGRU/temporal-attention architecture from notebook Cell 11.

with_attention=False reproduces the Cell 12 attention ablation.
"""
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, sequence):
        scores = self.score(sequence)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence * weights, dim=1)
        return context, weights.squeeze(-1)


class CNNBiGRUAttention(nn.Module):
    def __init__(self, input_dim=576, hidden_dim=128, num_classes=10,
                 dropout=0.3, with_attention=True):
        super().__init__()
        self.with_attention = with_attention
        self.input_norm = nn.LayerNorm(input_dim)
        self.bigru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        temporal_dim = hidden_dim * 2
        self.attention = TemporalAttention(temporal_dim, 64) if with_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(temporal_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.input_norm(x)
        sequence, _ = self.bigru(x)
        if self.with_attention:
            context, weights = self.attention(sequence)
        else:
            context = sequence.mean(dim=1)
            weights = torch.full(
                (sequence.size(0), sequence.size(1)),
                1.0 / sequence.size(1), device=sequence.device
            )
        logits = self.classifier(context)
        return (logits, weights) if return_attention else logits
