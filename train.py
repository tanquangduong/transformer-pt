from transformer.train_engine import train
from transformer.utils import load_config

if __name__ == "__main__":
    config_file_path = "config.json"
    config = load_config(config_file_path)
    train(config)
            