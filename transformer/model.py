import torch.nn as nn
from transformer.layers import InputEmbedding, PositionalEncoding, Projection
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
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
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_src = embed_src
        self.embed_tgt = embed_tgt
        self.pos_src = pos_src
        self.pos_tgt = pos_tgt
        self.projection = projection

    def encode(self, src, mask_src=None):
        # src: [batch_size, seq_len]
        # mask_src: [batch_size, 1, seq_len, seq_len]
        src = self.embed_src(src)
        src = self.pos_src(src)
        return self.encoder(src, mask_src)
    
    def decode(self, tgt, encoder_output, mask_src=None, mask_tgt=None):
        # tgt: [batch_size, seq_len]
        # encoder_output: [batch_size, seq_len, d_model]
        # mask_src: [batch_size, 1, seq_len, seq_len]
        # mask_tgt: [batch_size, 1, seq_len, seq_len]
        tgt = self.embed_tgt(tgt)
        tgt = self.pos_tgt(tgt)
        return self.decoder(tgt, encoder_output, mask_src, mask_tgt)
    
    def project(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.projection(x) # [batch_size, seq_len, vocab_size]
