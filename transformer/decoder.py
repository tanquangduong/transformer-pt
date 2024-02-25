import torch.nn as nn
from transformer.layer import FeedForward, MultiHeadAttention, ResidualConnection, LayerNorm

class DecoderLayer(nn.Module):
    """
    A decoder layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    self_attention (MultiHeadAttention): The multi-head self-attention mechanism.
    encoder_decoder_attention (MultiHeadAttention): The multi-head attention mechanism that connects the encoder and decoder.
    feed_forward (FeedForward): The feed-forward network.
    dropout (float): The dropout rate.
    """

    # The constructor takes the input/output dimensionality, the self-attention mechanism,
    # the encoder-decoder attention mechanism, the feed-forward network, and the dropout rate as parameters.
    def __init__(self, d_model: int, self_attention: MultiHeadAttention, encoder_decoder_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()  # Call the constructor of the parent class.
        
        # Store the self-attention mechanism, the encoder-decoder attention mechanism, and the feed-forward network.
        self.self_attention_engine = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.feed_forward = feed_forward
        
        # Initialize the residual connections.
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    # The forward method is called when we pass input data into this layer.
    def forward(self, x, encoder_output, mask_src=None, mask_tgt=None):
        # x: [batch_size, seq_len, d_model]
        # encoder_output: [batch_size, seq_len, d_model]
        
        # First, it applies the self-attention mechanism to the input inside a residual connection.
        # The mask_tgt is used to prevent the decoder from looking at future tokens.
        x = self.residual_connections[0](x, lambda x: self.self_attention_engine(x, x, x, mask_tgt))
        
        # Then, it applies the encoder-decoder attention mechanism to the result and the encoder output inside another residual connection.
        # The mask_src is used to prevent the decoder from looking at padding tokens.
        x = self.residual_connections[1](x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, mask_src))
        
        # Finally, it passes the result through the feed-forward network inside a third residual connection.
        # The output shape is [batch_size, seq_len, d_model].
        x = self.residual_connections[2](x, self.feed_forward)
        
        return x

class Decoder(nn.Module):
    """
    A decoder for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    decoder_layer (DecoderLayer): The type of layer to use in the decoder.
    num_layers (int): The number of layers in the decoder.
    """

    # The constructor takes the input/output dimensionality, the type of layer, and the number of layers as parameters.
    def __init__(self, d_model: int, decoder_layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()  # Call the constructor of the parent class.
        
        # Initialize the layers of the decoder.
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
        # Initialize the layer normalization.
        self.norm = LayerNorm(d_model)
    
    # The forward method is called when we pass input data into this module.
    def forward(self, x, encoder_output, mask_src=None, mask_tgt=None):
        # x: [batch_size, seq_len, d_model]
        
        # It passes the input and the encoder output through each layer in turn.
        for layer in self.layers:
            x = layer(x, encoder_output, mask_src, mask_tgt)
        
        # Finally, it applies layer normalization to the output of the last layer.
        # The output shape is [batch_size, seq_len, d_model].
        return self.norm(x)