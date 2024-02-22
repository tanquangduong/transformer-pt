import torch.nn as nn
from transformer.layer import FeedForward, MultiHeadAttention, ResidualConnection, LayerNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )
    
    def forward(self, x, mask_scr=None):
        # x: [batch_size, seq_len, d_model]
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask_scr))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, encoder_layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask_scr=None):
        # x: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x, mask_scr)
        return self.norm(x)
