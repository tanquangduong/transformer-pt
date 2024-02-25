import torch
from torch.utils.data import Dataset
from transformer.mask import create_encoder_mask, create_decoder_mask

# This class is used to preprocess the data for the transformer model.
# It inherits from the PyTorch Dataset class.
class DataPreprocessor(Dataset):
    """
    A data preprocessor for a transformer model.

    Parameters:
    dataset: The dataset to preprocess.
    tokenizer_src: The tokenizer for the source language.
    tokenizer_tgt: The tokenizer for the target language.
    language_src: The source language.
    language_tgt: The target language.
    seq_len: The maximum sequence length.
    """

    # The constructor takes the dataset, the tokenizers, the languages, and the sequence length as parameters.
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, language_src, language_tgt, seq_len):
        super().__init__()  # Call the constructor of the parent class.
        
        # Store the parameters.
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.language_src = language_src
        self.language_tgt = language_tgt
        self.seq_len = seq_len

        # Get the token ids for the special tokens.
        self.sos_token_id = torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token_id = torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token_id = torch.tensor([self.tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    # The __len__ method returns the number of items in the dataset.
    def __len__(self):
        return len(self.dataset)

    # The __getitem__ method returns the preprocessed data for the item at the given index.
    def __getitem__(self, idx):
        # Get the item from the dataset.
        item = self.dataset[idx]
        
        # Get the source and target texts from the item.
        text_src = item['translation'][self.language_src]
        text_tgt = item['translation'][self.language_tgt]

        # Tokenize the source and target texts.
        encoder_token_ids = torch.tensor(self.tokenizer_src.encode(text_src).ids, dtype=torch.int64)
        decoder_token_ids = torch.tensor(self.tokenizer_tgt.encode(text_tgt).ids, dtype=torch.int64)

        # Calculate the number of padding tokens needed for the encoder and decoder inputs.
        encoder_padding_num = self.seq_len - len(encoder_token_ids) - 2 # 2 for [SOS] and [EOS]
        decoder_padding_num = self.seq_len - len(decoder_token_ids) - 1 # 1: [SOS], [EOS] for decoder's input, target respectively

        # Create the encoder and decoder inputs and the decoder target.
        encoder_input_ids = torch.cat(
            [
                self.sos_token_id,
                encoder_token_ids,
                self.eos_token_id,
                self.pad_token_id.repeat(encoder_padding_num),
            ]
        )
        decoder_input_ids = torch.cat(
            [
                self.sos_token_id,
                decoder_token_ids,
                self.pad_token_id.repeat(decoder_padding_num),
            ]
        )
        decoder_target_ids = torch.cat(
            [
                decoder_token_ids,
                self.eos_token_id,
                self.pad_token_id.repeat(decoder_padding_num),
            ]
        ) # (seq_len)

        # Create the masks for the encoder and decoder inputs.
        encoder_mask = create_encoder_mask(encoder_input_ids, self.pad_token_id) # (1, 1, seq_len)
        decoder_mask = create_decoder_mask(decoder_input_ids, self.pad_token_id, self.seq_len) # (1, seq_len, seq_len)

        # Return the preprocessed data.
        output = {
            "text_src": text_src,
            "text_tgt": text_tgt,
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_target_ids": decoder_target_ids,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }
        return output