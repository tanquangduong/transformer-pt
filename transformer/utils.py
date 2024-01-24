import torch
import torch.nn as nn

# Our 'transformer' package imports
from transformer.layers import (
    InputEmbedding,
    PositionalEncoding,
    Projection,
    MultiHeadAttention,
    FeedForward,
)
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.model import Transformer

import os
from pathlib import Path
from tqdm import tqdm
import warnings

# HuggingFace imports
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def create_tranformer_model(
    vocab_size_src: int,
    vocab_size_tgt: int,
    d_model: int,
    num_layers: int,
    h: int,
    d_ff: int,
    dropout: float,
    seq_len: int,
) -> Transformer:
    # Initialize embedding layer
    embed_src = InputEmbedding(vocab_size_src, d_model)
    embed_tgt = InputEmbedding(vocab_size_tgt, d_model)

    # Initialize positional encoding layer
    pos_src = PositionalEncoding(d_model, seq_len, dropout)
    pos_tgt = PositionalEncoding(d_model, seq_len, dropout)

    # Initialize encoder
    self_attention_encoder = MultiHeadAttention(d_model, h, dropout)
    feed_forward_encoder = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(
        d_model, self_attention_encoder, feed_forward_encoder, dropout
    )
    encoder = Encoder(d_model, encoder_layer, num_layers)

    # Initialize decoder
    self_attention_decoder = MultiHeadAttention(d_model, h, dropout)
    encoder_decoder_attention = MultiHeadAttention(d_model, h, dropout)
    feed_forward_decoder = FeedForward(d_model, d_ff, dropout)
    decoder_layer = DecoderLayer(
        d_model,
        self_attention_decoder,
        encoder_decoder_attention,
        feed_forward_decoder,
        dropout,
    )
    decoder = Decoder(d_model, decoder_layer, num_layers)
    
    # Initialize projection layer
    projection = Projection(d_model, vocab_size_tgt)

    # Initialize transformer model
    transformer = Transformer(
        encoder,
        decoder,
        embed_src,
        embed_tgt,
        pos_src,
        pos_tgt,
        projection,
    )

    # Initialize model parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer

def get_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def get_tokenizer(tokenizer_path, dataset, language):
    def data_generator():
        for item in dataset['translation'][language]:
            yield item

    if Path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        )
        tokenizer.train_from_iterator(data_generator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
    return tokenizer
