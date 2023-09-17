import os
import pickle
from random import shuffle
import numpy as np

input_file_path = r"data\binary_levels\merged_levels_binary.txt"

# Read file line by line into a list
with open(input_file_path, 'r') as f:
    lines = f.readlines()
    
# Remove newline characters
lines = [line.strip() for line in lines]

# Shuffle the lines
shuffle(lines)

# Determine the split index
n = len(lines)
split_idx = int(n * 0.9)

# Create train and validation datasets
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

# Convert lists back to strings
train_data = ''.join(train_lines)
val_data = ''.join(val_lines)

# Combine train and validation data for vocabulary building
combined_data = train_data + val_data

# Get all the unique characters that occur in this text
chars = sorted(list(set(combined_data)))  # Changed to `combined_data`
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# Encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save the meta information as well
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
