import json
import torch
import torch.nn as nn
import time
from functools import wraps

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
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def create_tranformer_model(config, vocab_size_src, vocab_size_tgt) -> Transformer:
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    h = config["h"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]
    seq_len = config["seq_len"]

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


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def get_dataset(config):
    dataset_name = config["dataset_name"]
    lang_src = config["language_source"]
    lang_tgt = config["language_target"]
    language_pair = f"{lang_src}-{lang_tgt}"
    split = config["split"]
    raw_dataset = load_dataset(dataset_name, language_pair, split=split)
    return raw_dataset


def get_tokenizer(config, dataset, language):
    tokenizer_name = config["tokenizer_name"]
    tokenizer_path = f"{tokenizer_name}{language}.json"
    if Path.exists(Path(tokenizer_path)):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        )
        tokenizer.train_from_iterator(
            (item[language] for item in dataset["translation"]), trainer=trainer
        )
        tokenizer.save(tokenizer_path)
    return tokenizer


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


@timer
def calculate_max_lengths(dataset, tokenizer_src, tokenizer_tgt, config):
    max_src_len = max(
        len(tokenizer_src.encode(item["translation"][config["language_source"]]).ids)
        for item in dataset
    )
    max_tgt_len = max(
        len(tokenizer_tgt.encode(item["translation"][config["language_target"]]).ids)
        for item in dataset
    )
    return max_src_len, max_tgt_len


def get_checkpoint_path(config):
    model_dir = config["model_dir"]
    checkpoints = os.listdir(model_dir)
    checkpoints.sort()
    if len(checkpoints) == 0:
        warnings.warn("No checkpoints found")
        return None
    if config["preload"] == "latest":
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    elif config["preload"]:
        checkpoint_path = os.path.join(model_dir, config["preload"])
    else:
        checkpoint_path = None
    return checkpoint_path


def create_checkpoint_path(config, epoch):
    model_dir = config["model_dir"]
    checkpoint_basename = config["model_name"]
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_basename}{epoch}.pt")
    return str(checkpoint_path)

def create_causal_mask(size):
    return (
        torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
    )  # (1, size, size)
