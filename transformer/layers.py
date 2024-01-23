import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x): # x: [batch_size, seq_len]
        return self.embedding(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        base = 10000.0 ** (-1.0 / d_model)
        div_term = torch.pow(base, torch.arange(0, d_model, 2).float())

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): # x(embeded sequence): [batch_size, seq_len, d_model]
        x = x + self.pe.requires_grad_(False)
        return self.dropout(x) # [batch_size, seq_len, d_model]

class LayerNorm(nn.Module):
    def __init__(self, feature_len: int, eps: float=1e-6) -> None: 
        super().__init__()
        self.para_mul = nn.Parameter(torch.ones(feature_len))
        self.para_bias = nn.Parameter(torch.zeros(feature_len))
        self.eps = eps
    
    def forward(self, x): # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.para_mul * (x - mean) / (std + self.eps) + self.para_bias # [batch_size, seq_len, d_model]