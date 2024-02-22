import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.text as metrics
from transformer.mask import create_causal_mask

from tqdm import tqdm
from transformer.utils import (
    create_tranformer_model,
    get_checkpoint_path,
    create_checkpoint_path,
)
from transformer.utils import preprocessing_data


def train(config):
    # assign the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # initiate tensorboard writer
    logs = SummaryWriter(config["log_dir"])

    # check model folder exists, if not create one to save the model weights
    if not os.path.exists(config["model_dir"]):
        os.mkdir("models")

    # Load and preprocess the dataset
    (
        train_dataloader,
        val_dataloader,
        _,
        tokenizer_src,
        tokenizer_tgt,
    ) = preprocessing_data(config)

    # get vocab size for source and target language
    vocab_size_src = tokenizer_src.get_vocab_size()
    vocab_size_tgt = tokenizer_tgt.get_vocab_size()

    # Create the model
    model = create_tranformer_model(config, vocab_size_src, vocab_size_tgt).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], eps=1e-9
    )
    initial_epoch = 0
    global_step = 0

    # get the checkpoint path if exists
    checkpoint_path = get_checkpoint_path(config)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        initial_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
    else:
        print("No checkpoint found")

    # define the loss function
    loss_function = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # train loop
    for epoch in range(initial_epoch, config["epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch_size, 1, seq_len, seq_len)

            # pass the inputs through the model
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )  # (batch_size, seq_len, d_model)
            projection_output = model.project(
                decoder_output
            )  # (batch_size, seq_len, vocab_size)

            # Load target/label sequences
            decoder_target = batch["decoder_target"].to(device)  # (batch_size, seq_len)

            # Calculate loss
            loss = loss_function(
                projection_output.view(-1, vocab_size_tgt), decoder_target.view(-1)
            )
            batch_iterator.set_postfix({"Training Loss": loss.item()})

            # log the loss to tensorboard
            logs.add_scalar("Training Loss", loss.item(), global_step)
            logs.flush()

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        evaluation_step(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            global_step,
            logs,
            config["num_eval_samples"],
            device,
        )

        # Save model checkpoint
        model_checkpoint_path = create_checkpoint_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_checkpoint_path,
        )

def model_inference(model, encoder_input, encoder_mask, sos_id, eos_id, seq_len, device):
    # It is important to know that the sequence length in decoder_input will be the sequence length of the decoder output. For training, sequence length of decoder input == seq_len == sequence length of encoder input/output. However, for inference, the sequence length of decoder input will be increased by 1 at each step until the model predicts the end of sequence token. Therefore, the sequence length of decoder input will be different from the sequence length of encoder input/output.
    
    encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
    decoder_input = torch.tensor([[sos_id]]).type_as(encoder_input).to(device) # (1, 1) 

    while True:
        if decoder_input.shape[1] == seq_len:
            break
        decoder_mask = create_causal_mask(decoder_input.shape[1]).type_as(encoder_mask).to(device)
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


def evaluation_step(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, global_step, logs, num_eval_samples, device):

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

            source_text = batch["text_src"][0]
            target_text = batch["text_tgt"][0]
            predicted_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            target_texts.append(target_text)
            predicted_texts.append(predicted_text)

            print(f"Source: {source_text}")
            print(f"Target: {target_text}")
            print(f"Predicted: {predicted_text}")
            print("--------------------------------------------------")

            if len(source_texts) == num_eval_samples:
                print("*************** EVALUATION COMPLETED - GO TO NEXT EPOCH ***************")
                break
    
    if logs:

        # calculate the BLEU score
        bleu_metric = metrics.BLEUScore()
        bleu_score = bleu_metric(predicted_texts, target_texts)
        logs.add_scalar("Validation BLEU Score", bleu_score, global_step)
        logs.flush()

        # calculate the word error rate
        wer_metric  = metrics.WordErrorRate()
        wer_score = wer_metric(predicted_texts, target_texts)
        logs.add_scalar("Validation Word Error Rate", wer_score, global_step)
        logs.flush()

        # calculate character error rate
        cer_metric = metrics.CharErrorRate()
        cer_score = cer_metric(predicted_texts, target_texts)
        logs.add_scalar("Validation Character Error Rate", cer_score, global_step)
        logs.flush()

            

