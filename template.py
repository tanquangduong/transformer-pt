print("decoder_input_ids:", decoder_input_ids)
print("decoder_input_ids shape:", decoder_input_ids.shape)
print("\n")

# Create padding mask
padding_mask = create_padding_mask(decoder_input_ids, pad_token_id) # (1, seq_len)
print("padding_mask: \n", padding_mask)
print("padding_mask shape:", padding_mask.shape)
print("\n")

# Create a causal mask 
causal_mask = create_causal_mask(seq_len)
print('causal_mask: \n', causal_mask)
print('causal_mask shape:', causal_mask.shape)
print("\n")

# Create decoder mask
decoder_mask = create_decoder_mask(decoder_input_ids, pad_token_id, seq_len)
print('decoder_mask: \n', decoder_mask)
print('decoder_mask shape:', decoder_mask.shape)