import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from transformer.utils import (
    create_tranformer_model,
    get_checkpoint_path,
    create_checkpoint_path,
)
from transformer.dataset import preprocessing_data
from transformer.evaluation_engine import evaluation_step


def train(config):
    # assign the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # initiate tensorboard writer
    writer = SummaryWriter(config["log_dir"])

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
            writer.add_scalar("Training Loss", loss.item(), global_step)
            writer.flush()

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
            writer,
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
