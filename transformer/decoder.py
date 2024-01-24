import torch.nn as nn
from transformer.layers import FeedForward, MultiHeadAttention, ResidualConnection, LayerNorm

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, self_attention: MultiHeadAttention, encoder_decoder_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None: # encoder_decoder_attention in other words is cross attention between encoder and decoder
        super().__init__()
        self.self_attention_engine = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, mask_src=None, mask_tgt=None):
        # x: [batch_size, seq_len, d_model]
        # encoder_output: [batch_size, seq_len, d_model]
        x = self.residual_connections[0](x, lambda x: self.self_attention_engine(x, x, x, mask_tgt)) # here we use mask_tgt because we want to prevent decoder from looking at future tokens
        x = self.residual_connections[1](x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, mask_src)) # here we use mask_src because we want to prevent decoder from looking at padding tokens
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):

    def __init__(self, d_model: int, decoder_layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, mask_src=None, mask_tgt=None):
        # x: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x, encoder_output, mask_src, mask_tgt)
        return self.norm(x)