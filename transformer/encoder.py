import torch.nn as nn
from transformer.layer import FeedForward, MultiHeadAttention, ResidualConnection, LayerNorm


class EncoderLayer(nn.Module):
    """
    An encoder layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    self_attention (MultiHeadAttention): The multi-head self-attention mechanism.
    feed_forward (FeedForward): The feed-forward network.
    dropout (float): The dropout rate.
    """

    # The constructor takes the input/output dimensionality, the self-attention mechanism,
    # the feed-forward network, and the dropout rate as parameters.
    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()  # Call the constructor of the parent class.
        
        # Store the self-attention mechanism and the feed-forward network.
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        
        # Initialize the residual connections.
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )
    
    # The forward method is called when we pass input data into this layer.
    def forward(self, x, mask_scr=None):
        # x: [batch_size, seq_len, d_model]
        
        # First, it applies the self-attention mechanism to the input inside a residual connection.
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask_scr))
        
        # Then, it passes the result through the feed-forward network inside another residual connection.
        x = self.residual_connections[1](x, self.feed_forward)
        
        # The output shape is [batch_size, seq_len, d_model].
        return x

class Encoder(nn.Module):
    """
    An encoder for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    encoder_layer (EncoderLayer): The type of layer to use in the encoder.
    num_layers (int): The number of layers in the encoder.
    """

    # The constructor takes the input/output dimensionality, the type of layer, and the number of layers as parameters.
    def __init__(self, d_model: int, encoder_layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()  # Call the constructor of the parent class.
        
        # Initialize the layers of the encoder.
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
        # Initialize the layer normalization.
        self.norm = LayerNorm(d_model)
    
    # The forward method is called when we pass input data into this module.
    def forward(self, x, mask_scr=None):
        # x: [batch_size, seq_len, d_model]
        
        # It passes the input through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask_scr)
        
        # Finally, it applies layer normalization to the output of the last layer.
        # The output shape is [batch_size, seq_len, d_model].
        return self.norm(x)
