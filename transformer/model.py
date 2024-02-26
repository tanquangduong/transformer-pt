import torch.nn as nn
from transformer.layer import InputEmbedding, PositionalEncoding, Projection
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
    """
    This class represents a Transformer model, which is a type of sequence-to-sequence model
    that uses self-attention mechanisms. It consists of an encoder, a decoder, source and target
    input embeddings, source and target positional encodings, and a projection layer.

    The Transformer model is a subclass of the PyTorch Module class.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        embed_src: InputEmbedding,
        embed_tgt: InputEmbedding,
        pos_src: PositionalEncoding,
        pos_tgt: PositionalEncoding,
        projection: Projection,
    ) -> None:
        """
        Initializes the Transformer model with the given encoder, decoder, source and target
        input embeddings, source and target positional encodings, and projection layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_src = embed_src
        self.embed_tgt = embed_tgt
        self.pos_src = pos_src
        self.pos_tgt = pos_tgt
        self.projection = projection

    def encode(self, src, mask_src=None):
        """
        Encodes the source sequence.

        Args:
            src: The source sequence, a tensor of shape [batch_size, seq_len].
            mask_src: An optional mask for the source sequence, a tensor of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            The encoded source sequence.
        """
        src = self.embed_src(src)
        src = self.pos_src(src)
        return self.encoder(src, mask_src)
    
    def decode(self, tgt, encoder_output, mask_src=None, mask_tgt=None):
        """
        Decodes the target sequence.

        Args:
            tgt: The target sequence, a tensor of shape [batch_size, seq_len].
            encoder_output: The output of the encoder, a tensor of shape [batch_size, seq_len, d_model].
            mask_src: An optional mask for the source sequence, a tensor of shape [batch_size, 1, seq_len, seq_len].
            mask_tgt: An optional mask for the target sequence, a tensor of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            The decoded target sequence.
        """
        tgt = self.embed_tgt(tgt)
        tgt = self.pos_tgt(tgt)
        return self.decoder(tgt, encoder_output, mask_src, mask_tgt)
    
    def project(self, x):
        """
        Projects the output of the decoder to the vocabulary size.

        Args:
            x: The output of the decoder, a tensor of shape [batch_size, seq_len, d_model].

        Returns:
            The projected output, a tensor of shape [batch_size, seq_len, vocab_size].
        """
        return self.projection(x)