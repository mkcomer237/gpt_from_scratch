"""Bigram Model."""


import torch  
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import math


torch.manual_seed(1337)
# Larger batch size is faster (when using gpu at least)
val_pct = 0.1
batch_size = 128 # Number of independent sequences yt process in parallel
block_size = 8 # Maximum context length for the predictions 
learning_rate = 0.0003
num_epochs = 3
device = "cpu" # cuda, mps, or cpu
n_embed = 32



torch.manual_seed(1337)
batch_size = 4  # Number of independent sequences yt process in parallel
block_size = 8  # Maximum context length for the predictions 


# create a mapping from characters to integers and back


# Simple encoder and decoder functions
def encode(input_string, stoi):
    return [stoi[char] for char in input_string]


def decode(input_tokens, itos):
    return ''.join([itos[token] for token in input_tokens])


def tokenize_data(filepath):
    """Tokenize the raw data using one number per character."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Tokenize the data 
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    return data, vocab_size, stoi, itos


def train_test_split(data, val_pct):
    """Train test split on the data."""

    n = int(val_pct*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(split, train_data, val_data): # train or validation split
    """Generate a small batch of data from inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # batch_size random sequence starting points
    # print("Random starting points for each block: ", ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """ Forward pass. 
        
        This is an extremely simple model currently, where the embedding is used
        directly as an input into softmax to create an array of probabilities.  So the 
        embedding dimension must be equal to the vocab size since the emedding values itself
        are just the predictions.  So the model is taking each character and determining what
        the most likely next character is, trained on the offset by one x and y values."""
        # idx and targets are both (B, T) tensor of integers
        # We are ONLY using the embedding as the logits directly.  

        B, T = idx.shape

        # Add a linear layer to go from togken embeddings to logist now
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) - (Batch (4), Time (8), Channel(n_embed))
        position_embeddings = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        # Add position and token embeddings together (broadcasted over batches)
        x = token_embeddings + position_embeddings # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # Evaluate the loss (compare logits to the next character (targets))
        if targets == None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Stack the time pieces for each batch on top of each other batch
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens on top of the existing T tokens."""
        idx_output = idx
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # Keep the maximum to the block size (for position encodings)
            if idx.shape[1] > block_size:
                idx = idx[:, -block_size:]
            idx_output = torch.cat((idx_output, idx_next), dim=1) # (B, T+1)
        
        return idx_output


def get_optimization_details(model):

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, scheduler



def train(model, train_data, val_data):

    # Set up the model and optimizer
    
    batch_starts = np.arange(0, len(train_data)-batch_size, batch_size)
    batch_starts_val = np.arange(0, len(val_data)-batch_size, batch_size)

    optimizer, scheduler = get_optimization_details(model)

    # Note we use math.ceil for cases where the data to batch size division isn't a clean number
    n_steps = math.ceil(len(train_data)/batch_size)

    # Iterate through epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        model.train()

        # initialize tqdm object
        progress_bar = tqdm(total=n_steps, desc=f"Epoch {epoch+1}/{num_epochs}")

        batch_losses = []
        
        # Iterate through batches
        for batch_start in batch_starts:
            xb, yb = get_batch(train_data[batch_start:], train_data, val_data)
            xb = xb.to(device)
            yb = yb.to(device)
            # forward pass 
            logits, loss = model(xb, yb)
            
            batch_losses.append(loss.item())
            
            # update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            # Backward pass 
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
        total_loss = sum(batch_losses)/len(batch_losses)
        
        # Update learning rate
        scheduler.step()

        # TODO: add in a batch eval process here
        model.eval()
              
        # Print out training loop stats
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Execution time: {epoch_time:.4f}")
        progress_bar.close()

def main():
    """ Main entry point of the app."""

    data, vocab_size, stoi, itos = tokenize_data("input.txt")

    print(data[:100])

    model = BigramLanguageModel(vocab_size).to(device)

    train_data, val_data = train_test_split(data, val_pct)

    train(model, train_data, val_data)

    # Use the (currently untrained) model to generate new characters
    idx = torch.tensor([[51, 39, 62, 0]]).to(device)
    print(decode(model.generate(idx, 200)[0].tolist(), itos))

if __name__ == "__main__":
    """ This is executed when run from the command line."""
    main()

