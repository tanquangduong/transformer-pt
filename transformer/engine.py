import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.text as metrics
from transformer.mask import create_causal_mask, create_encoder_mask

from tqdm import tqdm
from transformer.utils import (
    create_tranformer_model,
    get_checkpoint_path,
    create_checkpoint_path,
)
from transformer.utils import preprocessing_data


def train(config):
    """
    This function trains a Transformer model for machine translation.

    Args:
        config (dict): A configuration dictionary containing parameters for training, such as the number of epochs, learning rate, sequence length, etc.

    """
    # Assign the device for computation (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Initiate TensorBoard writer for logging training metrics
    logs = SummaryWriter(config["log_dir"])

    # Check if model directory exists, if not create one to save the model weights
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

    # Get vocab size for source and target language
    vocab_size_src = tokenizer_src.get_vocab_size()
    vocab_size_tgt = tokenizer_tgt.get_vocab_size()

    # Create the Transformer model
    model = create_tranformer_model(config, vocab_size_src, vocab_size_tgt).to(device)

    # Initialize the optimizer (Adam in this case)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], eps=1e-9
    )
    initial_epoch = 0
    global_step = 0

    # Get the checkpoint path if exists
    checkpoint_path = get_checkpoint_path(config)

    # If a checkpoint exists, load the model and optimizer states and continue training from the last epoch
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        initial_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
    else:
        print("No checkpoint found")

    # Define the loss function (CrossEntropyLoss in this case)
    loss_function = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # Training loop
    for epoch in range(initial_epoch, config["epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")

        # Iterate over each batch in the training data
        for batch in batch_iterator:
            # Move the encoder and decoder inputs and masks to the device
            encoder_input = batch["encoder_input_ids"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["decoder_input_ids"].to(device)  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (batch_size, 1, seq_len, seq_len)

            # Pass the inputs through the model
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)  # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output)  # (batch_size, seq_len, vocab_size)

            # Load target/label sequences
            decoder_target = batch["decoder_target_ids"].to(device)  # (batch_size, seq_len)

            # Calculate loss
            loss = loss_function(projection_output.view(-1, vocab_size_tgt), decoder_target.view(-1))
            batch_iterator.set_postfix({"Training Loss": loss.item()})

            # Log the loss to TensorBoard
            logs.add_scalar("Training Loss", loss.item(), global_step)
            logs.flush()

            # Perform backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of each epoch
        evaluation(
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

def infer_training(model, encoder_input, encoder_mask, sos_id, eos_id, seq_len, device):
    """
    Performs inference using the provided model.

    Args:
        model (nn.Module): The Transformer model to use for inference.
        encoder_input (Tensor): The input to the encoder, a tensor of shape (batch_size, seq_len).
        encoder_mask (Tensor): The mask for the encoder input, a tensor of shape (1, 1, seq_len).
        sos_id (int): The ID of the start of sequence token.
        eos_id (int): The ID of the end of sequence token.
        seq_len (int): The length of the sequence.
        device (torch.device): The device to perform computations on.

    Returns:
        Tensor: The output of the decoder, a tensor of shape (seq_len).

    Note:
        The sequence length in decoder_input will be the sequence length of the decoder output. For training, sequence length of decoder input == seq_len == sequence length of encoder input/output. However, for inference, the sequence length of decoder input will be increased by 1 at each step until the model predicts the end of sequence token. Therefore, the sequence length of decoder input will be different from the sequence length of encoder input/output.
    """
    
    # Encode the input using the model
    encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
    
    # Initialize the decoder input with the start of sequence token
    decoder_input = torch.tensor([[sos_id]]).type_as(encoder_input).to(device) # (1, 1) 

    # Loop until the sequence length of the decoder input equals the sequence length of the encoder input
    while True:
        if decoder_input.shape[1] == seq_len:
            break

        # Create a causal mask for the decoder input
        decoder_mask = create_causal_mask(decoder_input.shape[1]).type_as(encoder_mask).to(device)
        
        # Decode the encoder output using the model
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # select the last token from the seq_len dimension
        last_token_output = decoder_output[:, -1, :]

        # project the output to the target vocab size
        projection_output = model.project(last_token_output)

        _, predicted_token = torch.max(projection_output, dim=1) # predicted_token is the indice of the max value in projection_output, meaning the id in vocabulary
        decoder_input = torch.cat([decoder_input, predicted_token.unsqueeze(0).type_as(encoder_input).to(device)], dim=1) 

        # Break the loop if the predicted token is the end of sequence token
        if predicted_token == eos_id:
            break

    # Return the decoder input, removing the batch dimension
    return decoder_input.squeeze(0)


def evaluation(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, global_step, logs, num_eval_samples, device):
    """
    Performs an evaluation step on the validation data.

    Args:
        model (nn.Module): The Transformer model to use for inference.
        val_dataloader (DataLoader): The DataLoader for the validation data.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        seq_len (int): The length of the sequence.
        global_step (int): The current global step in the training process.
        logs (SummaryWriter): The TensorBoard writer.
        num_eval_samples (int): The number of evaluation samples.
        device (torch.device): The device to perform computations on.

    """
    # Set the model to evaluation mode
    model.eval()
    
    # Get the IDs for the start and end of sequence tokens
    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    
    # Initialize lists to store the source texts, target texts, and predicted texts
    source_texts = []
    target_texts = []
    predicted_texts = []

    # Disable gradient calculations
    with torch.no_grad():
        # Iterate over the validation data
        for batch in val_dataloader:
            # Get the encoder input and mask from the batch and move them to the device
            encoder_input = batch["encoder_input_ids"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            # Assert that the batch size is 1 for evaluation
            assert encoder_input.shape[0] == encoder_mask.shape[0] == 1, "Batch size must be 1 for evaluation"

            # Perform inference using the model
            model_output = infer_training(model, encoder_input, encoder_mask, sos_id, eos_id, seq_len, device)

            # Get the source text, target text, and predicted text from the batch
            source_text = batch["text_src"][0]
            target_text = batch["text_tgt"][0]
            predicted_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # Append the texts to their respective lists
            source_texts.append(source_text)
            target_texts.append(target_text)
            predicted_texts.append(predicted_text)

            # Print the source text, target text, and predicted text
            print(f"Source: {source_text}")
            print(f"Target: {target_text}")
            print(f"Predicted: {predicted_text}")
            print("--------------------------------------------------")

            # Break the loop if the number of evaluation samples has been reached
            if len(source_texts) == num_eval_samples:
                print("*************** EVALUATION COMPLETED - GO TO NEXT EPOCH ***************")
                break
    
    # If logs are provided, calculate and log the BLEU score, word error rate, and character error rate
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

def inference(src_text, model, tokenizer_src, tokenizer_tgt, seq_len, device):
    """
    Translates a source text using a transformer model.

    Parameters:
    src_text (str): The source text to translate.
    model (nn.Module): The transformer model to use for translation.
    tokenizer_src (Tokenizer): The tokenizer for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer for the target language.
    seq_len (int): The maximum sequence length.
    device (torch.device): The device to run the model on.

    Returns:
    str: The translated text.
    """

    # Set the model to evaluation mode.
    model.eval()

    # Disable gradient calculation.
    with torch.no_grad():

        # Tokenize the source text.
        tokenized_src_text = tokenizer_src.encode(src_text)

        # Calculate the number of padding tokens needed for the encoder input.
        encoder_padding_num = seq_len - len(tokenized_src_text.ids) - 2 # 2 for [SOS] and [EOS]

        # Get the IDs of the special tokens.
        sos_id = tokenizer_src.token_to_id("[SOS]")
        eos_id = tokenizer_src.token_to_id("[EOS]")
        pad_id = tokenizer_src.token_to_id("[PAD]")

        # Create the encoder input by concatenating the special tokens and the tokenized source text.
        encoder_input = torch.cat(
            [
                torch.tensor([sos_id],  dtype=torch.int64),
                torch.tensor(tokenized_src_text.ids, dtype=torch.int64),
                torch.tensor([eos_id],  dtype=torch.int64),
                torch.tensor([pad_id] * encoder_padding_num,  dtype=torch.int64),
            ], dim=0
        ).to(device)

        # Create the encoder mask.
        encoder_mask = create_encoder_mask(encoder_input, pad_id).to(device)

        # Encode the input.
        encoder_output = model.encode(encoder_input, encoder_mask)

        # Initialize the decoder input with the start of sequence token.
        decoder_input = torch.tensor([[sos_id]]).type_as(encoder_input).to(device)

        # Loop until the sequence length is reached or the end of sequence token is predicted.
        while True:

            # Break the loop if the sequence length is reached.
            if decoder_input.shape[1] == seq_len:
                break

            # Create the decoder mask.
            decoder_mask = create_causal_mask(decoder_input.shape[1]).type_as(encoder_mask).to(device)

            # Decode the encoder output.
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

            # Select the last token from the sequence length dimension.
            last_token_output = decoder_output[:, -1, :]

            # Project the output to the target vocabulary size.
            projection_output = model.project(last_token_output)

            # Get the predicted token by finding the index of the maximum value in the projection output.
            _, predicted_token = torch.max(projection_output, dim=1)

            # Update the decoder input by appending the predicted token.
            decoder_input = torch.cat([decoder_input, predicted_token.unsqueeze(0).type_as(encoder_input).to(device)], dim=1) 

            # Break the loop if the predicted token is the end of sequence token.
            if predicted_token == eos_id:
                break
    
    # Decode the decoder input to get the translated text.
    translated_text = tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())
    
    return translated_text
