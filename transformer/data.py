import torch
from torch.utils.data import Dataset
from transformer.mask import create_encoder_mask, create_decoder_mask

class DataPreprocessor(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, language_src, language_tgt, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.language_src = language_src
        self.language_tgt = language_tgt
        self.seq_len = seq_len

        self.sos_token_id = torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token_id = torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token_id = torch.tensor([self.tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text_src = item['translation'][self.language_src]
        text_tgt = item['translation'][self.language_tgt]

        # Tokenize source and target sequences
        encoder_token_ids = torch.tensor(self.tokenizer_src.encode(text_src).ids, dtype=torch.int64)
        decoder_token_ids = torch.tensor(self.tokenizer_tgt.encode(text_tgt).ids, dtype=torch.int64)

        # Add special tokens
        encoder_padding_num = self.seq_len - len(encoder_token_ids) - 2 # 2 for [SOS] and [EOS]
        decoder_padding_num = self.seq_len - len(decoder_token_ids) - 1 # 1: [SOS], [EOS] for decoder's input, target respectively

        # Create fixed length sequences of token ids for encoder's and decoder's inputs
        encoder_input = torch.cat(
            [
                self.sos_token_id,
                encoder_token_ids,
                self.eos_token_id,
                self.pad_token_id.repeat(encoder_padding_num),
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token_id,
                decoder_token_ids,
                self.pad_token_id.repeat(decoder_padding_num),
            ]
        )
        decoder_target = torch.cat(
            [
                decoder_token_ids,
                self.eos_token_id,
                self.pad_token_id.repeat(decoder_padding_num),
            ]
        ) # (seq_len)

        # mask out the padding tokens in the encoder's input during attention calculation
        encoder_mask = create_encoder_mask(encoder_input, self.pad_token_id) # (1, 1, seq_len)

        # mask out the future tokens in the decoder's input
        decoder_mask = create_decoder_mask(decoder_input, self.pad_token_id, self.seq_len) # (1, seq_len, seq_len)

        output = {
            "text_src": text_src,
            "text_tgt": text_tgt,
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }
        return output

