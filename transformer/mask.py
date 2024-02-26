import torch


def create_encoder_mask(encoder_input_ids, pad_token_id):
    """
    Creates a mask for the encoder input IDs.

    Args:
        encoder_input_ids (Tensor): The encoder input IDs.
        pad_token_id (int): The ID of the padding token.

    Returns:
        Tensor: A mask tensor of shape (1, 1, seq_len).
    """
    return (encoder_input_ids != pad_token_id).unsqueeze(0).unsqueeze(0).int()


def create_padding_mask(decoder_input_ids, pad_token_id):
    """
    Creates a padding mask for the decoder input IDs.

    Args:
        decoder_input_ids (Tensor): The decoder input IDs.
        pad_token_id (int): The ID of the padding token.

    Returns:
        Tensor: A mask tensor of shape (1, seq_len).
    """
    return (decoder_input_ids != pad_token_id).unsqueeze(0).int()


def create_causal_mask(seq_len):
    """
    Creates a causal mask for a sequence of a given length.

    Args:
        seq_len (int): The length of the sequence.

    Returns:
        Tensor: A mask tensor of shape (1, seq_len, seq_len).
    """
    return (torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int) == 0)


def create_decoder_mask(decoder_input_ids, pad_token_id, seq_len):
    """
    Creates a mask for the decoder input IDs, which is the logical AND of a padding mask and a causal mask.

    Args:
        decoder_input_ids (Tensor): The decoder input IDs.
        pad_token_id (int): The ID of the padding token.
        seq_len (int): The length of the sequence.

    Returns:
        Tensor: A mask tensor of shape (1, seq_len, seq_len).
    """
    padding_mask = create_padding_mask(decoder_input_ids, pad_token_id)
    causal_mask = create_causal_mask(seq_len)
    return padding_mask & causal_mask
