import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
    def __init__(self, d_model: int, eps: float=1e-6) -> None: 
        super().__init__()
        self.para_mul = nn.Parameter(torch.ones(d_model))
        self.para_bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x): # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.para_mul * (x - mean) / (std + self.eps) + self.para_bias # [batch_size, seq_len, d_model]
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): # x: [batch_size, seq_len, d_model]
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) # [batch_size, seq_len, d_model]
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer): # x: [batch_size, seq_len, d_model]
        return x + self.dropout(sublayer(self.norm(x))) # [batch_size, seq_len, d_model]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None: 
        # d_model: feature length of token
        # h: number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h

        # d_model % num_heads should be zero
        assert d_model % h == 0, "d_model % num_heads should be zero" 
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False) # parameter matrix for query W_Q
        self.w_k = nn.Linear(d_model, d_model, bias=False) # parameter matrix for key W_K
        self.w_v = nn.Linear(d_model, d_model, bias=False) # parameter matrix for value W_V
        self.w_o = nn.Linear(d_model, d_model, bias=False) # parameter matrix for output W_O
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query_k, key_k, value_k, d_k, mask=None, dropout=nn.Dropout):
        # query_k: [batch_size, h, seq_len, d_k]
        # key_k: [batch_size, h, seq_len, d_k]
        # value_k: [batch_size, h, seq_len, d_k]
        # mask: [batch_size, 1, seq_len, seq_len]

        attention_score = (query_k @ key_k.transpose(-2, -1)) / math.sqrt(d_k) # [batch_size, h, seq_len, seq_len]

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        
        attention_score = torch.softmax(attention_score, dim=-1) # [batch_size, h, seq_len, seq_len]
        attention_score = dropout(attention_score)

        return attention_score @ value_k, attention_score # [batch_size, h, seq_len, d_k], [batch_size, h, seq_len, seq_len]
    
    def forward(self, query, key, value, mask=None):
        # query: [batch_size, seq_len, d_model]
        # key: [batch_size, seq_len, d_model]
        # value: [batch_size, seq_len, d_model]
        # mask: [batch_size, 1, seq_len, seq_len]

        query_k = self.w_q(query) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        key_k = self.w_k(key) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        value_k = self.w_v(value) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]

        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k] -> [batch_size, h, seq_len, d_k]
        query_k = query_k.view(query_k.shape[0], query_k.shape[1], self.h, self.d_k).transpose(1, 2)
        key_k = key_k.view(key_k.shape[0], key_k.shape[1], self.h, self.d_k).transpose(1, 2)
        value_k = value_k.view(value_k.shape[0], value_k.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention  
        attention, _ = self.attention(query_k, key_k, value_k, self.d_k, mask, self.dropout)

        # Concatenate h heads
        # [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model]
        attention = attention.transpose(1, 2).contiguous().view(attention.shape[0], -1, self.d_model)

        return self.w_o(attention) # [batch_size, seq_len, d_model]

class Projection(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x): # x: [batch_size, seq_len, d_model]
        return self.projection(x) # [batch_size, seq_len, vocab_size]
        
class WordDecoder(nn.Module):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x): # x: [batch_size, seq_len, vocab_size]
        output = F.softmax(x, dim=-1) # Apply softmax to the last dimension
        top_token = torch.argmax(output, dim=-1) # Get the token with the highest probability

        # Iterate over the top_token tensor and decode each token in each sequence separately
        decoded_words = [[self.tokenizer.decode(t.item()) for t in sequence] for sequence in top_token]

        return decoded_words