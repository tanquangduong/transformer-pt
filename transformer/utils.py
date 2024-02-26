import json
import time
from functools import wraps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Our 'transformer' package imports
from transformer.layer import (
    InputEmbedding,
    PositionalEncoding,
    Projection,
    MultiHeadAttention,
    FeedForward,
)
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.model import Transformer
from transformer.data import DataPreprocessor

import os
from pathlib import Path
from tqdm import tqdm
import warnings

# HuggingFace imports
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def load_config(config_file_path):
    """
    Load a configuration file in JSON format.

    Parameters:
    config_file_path (str): The path to the configuration file.

    Returns:
    dict: A dictionary containing the configuration data.
    """
    # Open the file in read mode
    with open(config_file_path, "r") as f:
        # Load the JSON content of the file into a Python dictionary
        config = json.load(f)
    # Return the configuration data
    return config


def get_dataset(config):
    """
    Retrieve a specific dataset from HuggingFace's datasets library.

    Parameters:
    config (dict): A dictionary containing the configuration data. It should include:
        - "dataset_name": The name of the dataset to load.
        - "language_source": The source language code.
        - "language_target": The target language code.
        - "split": The specific split of the dataset to load (e.g., 'train', 'test').

    Returns:
    Dataset: The loaded dataset.
    """
    # Extract the dataset name, source language, target language, and split from the config
    dataset_name = config["dataset_name"]
    lang_src = config["language_source"]
    lang_tgt = config["language_target"]
    split = config["split"]

    # Construct the language pair string
    language_pair = f"{lang_src}-{lang_tgt}"

    # Load the dataset using the provided parameters
    raw_dataset = load_dataset(dataset_name, language_pair, split=split)

    # Return the loaded dataset
    return raw_dataset


# This function retrieves or trains a tokenizer based on the provided configuration, dataset, and language.
# If a tokenizer file already exists at the specified path, it loads the tokenizer from that file.
# Otherwise, it creates a new tokenizer, trains it on the provided dataset, and saves it to the specified path.
def get_tokenizer(config, dataset, language):
    """
    Retrieve or train a tokenizer based on the provided configuration, dataset, and language.

    Parameters:
    config (dict): A dictionary containing the configuration data. It should include "tokenizer_name".
    dataset (Dataset): The dataset to train the tokenizer on if necessary.
    language (str): The language to use for the tokenizer.

    Returns:
    Tokenizer: The loaded or trained tokenizer.
    """
    # Extract the tokenizer name from the config and construct the tokenizer file path
    tokenizer_name = config["tokenizer_name"]
    tokenizer_path = f"{tokenizer_name}{language}.json"

    # Check if a tokenizer file already exists at the specified path
    if Path.exists(Path(tokenizer_path)):
        # If it does, load the tokenizer from that file
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        # If it doesn't, create a new tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Create a trainer for the tokenizer
        trainer = trainers.WordLevelTrainer(
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        )

        # Train the tokenizer on the provided dataset
        tokenizer.train_from_iterator(
            (item[language] for item in dataset["translation"]), trainer=trainer
        )

        # Save the trained tokenizer to the specified path
        tokenizer.save(tokenizer_path)

    # Return the tokenizer
    return tokenizer

def preprocessing_data(config):
    """
    This function preprocesses the dataset for training, validation, and testing.
    It tokenizes the raw dataset, splits it into training, validation, and test sets,
    and then creates DataLoaders for each set.

    Args:
        config (dict): A configuration dictionary containing parameters for preprocessing.
            It should include:
            - "language_source": The source language for tokenization.
            - "language_target": The target language for tokenization.
            - "seq_len": The sequence length for the DataPreprocessor.
            - "batch_size": The batch size for the DataLoader.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): DataLoader for the training set.
            - val_dataloader (DataLoader): DataLoader for the validation set.
            - test_dataloader (DataLoader): DataLoader for the test set.
            - tokenizer_src (Tokenizer): Tokenizer for the source language.
            - tokenizer_tgt (Tokenizer): Tokenizer for the target language.
    """

    # Get the raw dataset
    raw_dataset = get_dataset(config)

    # Initialize tokenizers for the source and target languages
    tokenizer_src = get_tokenizer(config, raw_dataset, config["language_source"])
    tokenizer_tgt = get_tokenizer(config, raw_dataset, config["language_target"])

    # Calculate the sizes of the training, validation, and test sets
    train_ds_size = int(len(raw_dataset) * 0.7)
    val_ds_size = int(len(raw_dataset) * 0.2)
    test_ds_size = len(raw_dataset) - train_ds_size - val_ds_size

    # Split the raw dataset into training, validation, and test sets
    train_raw_dataset, val_raw_dataset, test_raw_dataset = random_split(
        raw_dataset, [train_ds_size, val_ds_size, test_ds_size]
    )

    # Preprocess the training, validation, and test sets
    train_ds = DataPreprocessor(
        train_raw_dataset,
        tokenizer_src,
        tokenizer_tgt,
        config["language_source"],
        config["language_target"],
        config["seq_len"],
    )
    val_ds = DataPreprocessor(
        val_raw_dataset,
        tokenizer_src,
        tokenizer_tgt,
        config["language_source"],
        config["language_target"],
        config["seq_len"],
    )
    test_ds = DataPreprocessor(
        test_raw_dataset,
        tokenizer_src,
        tokenizer_tgt,
        config["language_source"],
        config["language_target"],
        config["seq_len"],
    )

    # Create DataLoaders for the training, validation, and test sets
    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    )


def create_tranformer_model(config, vocab_size_src, vocab_size_tgt) -> Transformer:
    """
    This function creates a Transformer model with the given configuration and vocabulary sizes.

    Args:
        config (dict): A configuration dictionary containing parameters for the Transformer model.
            It should include:
            - "d_model": The dimension of the model.
            - "num_layers": The number of layers.
            - "h": The number of heads for the multi-head attention mechanism.
            - "d_ff": The dimension of the feed-forward network.
            - "dropout": The dropout rate.
            - "seq_len": The sequence length.
        vocab_size_src (int): The size of the source vocabulary.
        vocab_size_tgt (int): The size of the target vocabulary.

    Returns:
        Transformer: A Transformer model.
    """

    # Extract parameters from the configuration
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    h = config["h"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]
    seq_len = config["seq_len"]

    # Initialize the source and target embedding layers
    embed_src = InputEmbedding(vocab_size_src, d_model)
    embed_tgt = InputEmbedding(vocab_size_tgt, d_model)

    # Initialize the source and target positional encoding layers
    pos_src = PositionalEncoding(d_model, seq_len, dropout)
    pos_tgt = PositionalEncoding(d_model, seq_len, dropout)

    # Initialize the encoder with self-attention and feed-forward layers
    self_attention_encoder = MultiHeadAttention(d_model, h, dropout)
    feed_forward_encoder = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(
        d_model, self_attention_encoder, feed_forward_encoder, dropout
    )
    encoder = Encoder(d_model, encoder_layer, num_layers)

    # Initialize the decoder with self-attention, encoder-decoder attention, and feed-forward layers
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

    # Initialize the projection layer
    projection = Projection(d_model, vocab_size_tgt)

    # Initialize the Transformer model
    transformer = Transformer(
        encoder,
        decoder,
        embed_src,
        embed_tgt,
        pos_src,
        pos_tgt,
        projection,
    )

    # Initialize the model parameters with Xavier uniform distribution
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer

def timer(func):
    """
    This is a decorator function that calculates and prints the execution time of the decorated function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.
    """
    
    @wraps(func)  # This preserves the name, docstring, etc. of the original function
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time before executing the function

        result = func(*args, **kwargs)  # Execute the function and store the result

        end_time = time.time()  # Record the end time after executing the function

        # Calculate and print the execution time
        print(f"Execution time: {end_time - start_time} seconds")

        return result  # Return the result of the function

    return wrapper  # Return the decorated function


@timer
def calculate_max_lengths(dataset, tokenizer_src, tokenizer_tgt, config):
    """
    This function calculates the maximum lengths of the source and target sequences in the dataset.
    The function is decorated with the @timer decorator, which measures the execution time.

    Args:
        dataset (Dataset): The dataset to calculate the maximum lengths from.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        config (dict): A configuration dictionary containing parameters for the function.
            It should include:
            - "language_source": The source language.
            - "language_target": The target language.

    Returns:
        tuple: A tuple containing:
            - max_src_len (int): The maximum length of the source sequences.
            - max_tgt_len (int): The maximum length of the target sequences.
    """

    # Calculate the maximum length of the source sequences
    max_src_len = max(
        len(tokenizer_src.encode(item["translation"][config["language_source"]]).ids)
        for item in dataset
    )

    # Calculate the maximum length of the target sequences
    max_tgt_len = max(
        len(tokenizer_tgt.encode(item["translation"][config["language_target"]]).ids)
        for item in dataset
    )

    return max_src_len, max_tgt_len


def get_checkpoint_path(config):
    """
    This function retrieves the path of the checkpoint file based on the provided configuration.

    Args:
        config (dict): A configuration dictionary containing parameters for the function.
            It should include:
            - "model_dir": The directory where the model checkpoints are stored.
            - "preload": The checkpoint to load. If "latest", the latest checkpoint is loaded.
                If a filename is provided, that checkpoint is loaded.
                If False or None, no checkpoint is loaded.

    Returns:
        str or None: The path of the checkpoint file, or None if no checkpoint is to be loaded.
    """

    # Get the directory where the model checkpoints are stored
    model_dir = config["model_dir"]

    # List all files in the model directory and sort them
    checkpoints = os.listdir(model_dir)
    checkpoints.sort()

    # If there are no checkpoints, issue a warning and return None
    if len(checkpoints) == 0:
        warnings.warn("No checkpoints found")
        return None

    # If the configuration specifies to preload the latest checkpoint, get its path
    if config["preload"] == "latest":
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)

    # If the configuration specifies to preload a specific checkpoint, get its path
    elif config["preload"]:
        checkpoint_path = os.path.join(model_dir, config["preload"])

    # If the configuration does not specify to preload any checkpoint, return None
    else:
        checkpoint_path = None

    return checkpoint_path


def create_checkpoint_path(config, epoch):
    """
    This function creates the path for a checkpoint file based on the provided configuration and epoch number.

    Args:
        config (dict): A configuration dictionary containing parameters for the function.
            It should include:
            - "model_dir": The directory where the model checkpoints are stored.
            - "model_name": The base name of the checkpoint files.
        epoch (int): The epoch number.

    Returns:
        str: The path of the checkpoint file.
    """

    # Get the directory where the model checkpoints are stored
    model_dir = config["model_dir"]

    # Get the base name of the checkpoint files
    checkpoint_basename = config["model_name"]

    # Create the path of the checkpoint file
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_basename}{epoch}.pt")

    return str(checkpoint_path)

