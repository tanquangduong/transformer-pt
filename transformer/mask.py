import torch


def create_encoder_mask(encoder_input, pad_token_id):
    return (
        (encoder_input != pad_token_id).unsqueeze(0).unsqueeze(0).int()
    )  # (1, 1, seq_len)


def create_padding_mask(decoder_input, pad_token_id):
    return (decoder_input != pad_token_id).unsqueeze(0).int()  # (1, seq_len)


def create_causal_mask(seq_len):
    return (
        torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int) == 0
    )  # (1, seq_len, seq_len)


def create_decoder_mask(decoder_input, pad_token_id, seq_len):
    padding_mask = create_padding_mask(decoder_input, pad_token_id)
    causal_mask = create_causal_mask(seq_len)
    return padding_mask & causal_mask  # (1, seq_len, seq_len)
