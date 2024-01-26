import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from tqdm import tqdm
from transformer.utils import load_config, preprocessing_data, create_tranformer_model, get_checkpoint_path, create_causal_mask

def model_inference(model, encoder_input, encoder_mask, sos_id, eos_id, seq_len, device):
    # It is important to know that the sequence length in decoder_input will be the sequence length of the decoder output. For training, sequence length of decoder input == seq_len == sequence length of encoder input/output. However, for inference, the sequence length of decoder input will be increased by 1 at each step until the model predicts the end of sequence token. Therefore, the sequence length of decoder input will be different from the sequence length of encoder input/output.
    
    encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
    decoder_input = torch.tensor([[sos_id]]).type_as(encoder_input).to(device) # (1, 1) 

    while True:
        if decoder_input.shape[1] == seq_len:
            break
        decoder_mask = create_causal_mask(decoder_input.shape[1]).as_type(encoder_mask).to(device)
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # select the last token from the seq_len dimension
        last_token_output = decoder_output[:, -1, :]

        # project the output to the target vocab size
        projection_output = model.project(last_token_output)

        _, predicted_token = torch.max(projection_output, dim=1) # predicted_token is the indice of the max value in projection_output, meaning the id in vocabulary
        decoder_input = torch.cat([decoder_input, predicted_token.unsqueeze(0).type_as(encoder_input).to(device)], dim=1) 

        if predicted_token == eos_id:
            break
    
    return decoder_input.squeeze(0)


def evaluation_step(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, global_step, writer, num_eval_samples, device):

    model.eval()
    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    source_texts = []
    target_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            assert encoder_input.shape[0] == encoder_mask.shape[0] == 1, "Batch size must be 1 for evaluation"

            model_output = model_inference(model, encoder_input, encoder_mask, sos_id, eos_id, seq_len, device)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            predicted_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            target_texts.append(target_text)
            predicted_texts.append(predicted_text)

            print(f"Source: {source_text}")
            print(f"Target: {target_text}")
            print(f"Predicted: {predicted_text}")

            if len(source_texts) == num_eval_samples:
                print("*************** EVALUATION COMPLETED ***************")
                break
    
    if writer:

        # calculate the BLEU score
        bleu_metric = torchmetrics.BLEUScore()
        bleu_score = bleu_metric(predicted_texts, target_texts)
        writer.add_scalar("Validation BLEU Score", bleu_score, global_step)
        writer.flush()

        # calculate the word error rate
        wer_metric  = torchmetrics.WordErrorRate()
        wer_score = wer_metric(predicted_texts, target_texts)
        writer.add_scalar("Validation Word Error Rate", wer_score, global_step)
        writer.flush()

        # calculate character error rate
        cer_metric = torchmetrics.CharErrorRate()
        cer_score = cer_metric(predicted_texts, target_texts)
        writer.add_scalar("Validation Character Error Rate", cer_score, global_step)
        writer.flush()

            

