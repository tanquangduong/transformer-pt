import sys
sys.path.append('../')
import torch
from transformer.layer import MultiHeadAttention, FeedForward, LayerNorm
from transformer.encoder import EncoderLayer, Encoder

# Define configuration
d_model = 6 # feature dimension
h = 3 #  number of heads
dropout = 0.1 # dropout ratio
d_ff = 2048 # the dimension of the feed forward network
batch = 1 # batch size
seq_len = 4 # sequence length

num_layers = 3 # number of encoder layer

# Create an instance of the MultiHeadAttention and FeedForward classes
self_attention_engine = MultiHeadAttention(d_model, h, dropout)
feed_forward = FeedForward(d_model, d_ff, dropout)  

# Create an instance of the EncoderLayer class
encoder_layer = EncoderLayer(d_model, self_attention_engine, feed_forward, dropout)

# Create an instance of the Encoder class
encoder = Encoder(d_model, encoder_layer, num_layers)

# Create a random tensor to represent a batch of sequences
torch.manual_seed(68) # for reproducible result of random process
x = torch.rand(batch, seq_len, d_model)  

# Pass the tensor through the encoder
output = encoder(x)

print("Initial input tensor: \n", x)
print("Encoder Output: \n", output)
print("Encoder Output's shape: \n", output.shape)  