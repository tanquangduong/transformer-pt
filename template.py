import torch
from transformer.utils import create_tranformer_model
from transformer.utils import load_config, get_dataset, get_tokenizer
from transformer.utils import get_checkpoint_path
from transformer.engine import transformer_translates

# Assign the device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

## Get Config, dataset, tokenizers
config_file_path = "./config.json"

config = load_config(...)
dataset = get_dataset(...)
tokenizer_src = get_tokenizer(...)

tokenizer_tgt = get_tokenizer(config, dataset, config['language_target'])

vocab_size_src = tokenizer_src.get_vocab_size()
vocab_size_tgt = tokenizer_tgt.get_vocab_size()
seq_len = config["seq_len"]

# Create the Transformer model
model = create_tranformer_model(config, vocab_size_src, vocab_size_tgt).to(device)
# Load latest checkpoint
model_checkpoint = get_checkpoint_path(config)
print("model_checkpoint:", model_checkpoint)
state = torch.load(model_checkpoint)
# Assign latest checkpoint to transfromer
model.load_state_dict(state['model_state_dict'])

# Source text to translate
source_text = "Market prices are shaped by the balance of supply and demand."
# the correct translation could be: 
# "Les prix du marché sont déterminés par l'équilibre entre l'offre et la demande."

translated_text = transformer_translates(source_text, 
                                        model, 
                                        tokenizer_src, 
                                        tokenizer_tgt,
                                        seq_len,
                                        device)

print("Source text: \n", source_text)
print(">>>>")
print("Translated text: \n", translated_text)