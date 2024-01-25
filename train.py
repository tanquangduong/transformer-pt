import os
import torch
from transformer.utils import load_config, preprocessing_data, create_tranformer_model

def train(config):
    # assign the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # check model folder exists, if not create one to save the model weights
    if not os.path.exists(config["model_dir"]):
        os.mkdir("models")
    
    # Load and preprocess the dataset
    train_dataset, val_dataset, test_dataset, tokenizer_src, tokenizer_tgt = preprocessing_data(config)

    # Create the model
    


