import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformer.utils import create_causal_mask, get_dataset, get_tokenizer

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
        encoder_mask = (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)

        # mask out the padding tokens in the decoder's input
        padding_mask = (decoder_input != self.pad_token_id).unsqueeze(0).int() # (1, seq_len)

        # causal mask, also known as the look-ahead mask, used to mask out the future tokens in a sequence, making sure that the predictions for a given token only depend on the tokens that came before it.
        causal_mask = create_causal_mask(self.seq_len) # (1, seq_len, seq_len)


        output = {
            "text_src": text_src,
            "text_tgt": text_tgt,
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "encoder_mask": encoder_mask,
            "decoder_mask": padding_mask & causal_mask, # (1, seq_len, seq_len)
        }
        return output

def preprocessing_data(config):
    raw_dataset = get_dataset(config)

    # call tokenizer
    tokenizer_src = get_tokenizer(config, raw_dataset, config["language_source"])
    tokenizer_tgt = get_tokenizer(config, raw_dataset, config["language_target"])

    # split raw dataset: 70% train, 20% validation, 10% test
    train_ds_size = int(len(raw_dataset) * 0.7)
    val_ds_size = int(len(raw_dataset) * 0.2)
    test_ds_size = len(raw_dataset) - train_ds_size - val_ds_size

    train_raw_dataset, val_raw_dataset, test_raw_dataset = random_split(
        raw_dataset, [train_ds_size, val_ds_size, test_ds_size]
    )

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
