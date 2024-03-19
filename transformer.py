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
batch_size = 1024 # Number of independent sequences yt process in parallel
block_size = 64 # Maximum context length for the predictions 
# Set number of batches to randomly generate for each epoch
train_num_batches = int(10000*64/batch_size)
val_num_batches = int(1000*64/batch_size)
learning_rate = 0.00006
lr_decay = 0.95
num_epochs = 50
device = "cuda" # cuda, mps, or cpu
n_embed = 64

torch.manual_seed(1337)
# batch_size = 4  # Number of independent sequences yt process in parallel
# block_size = 8  # Maximum context length for the predictions 


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

    n = int((1-val_pct)*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(split, train_data, val_data): # train or validation split
    """Generate a small batch of data from inputs x and targets y."""
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be train or val")
    ix = torch.randint(len(data) - block_size, (batch_size,)) # batch_size random sequence starting points
    # print("Random starting points for each block: ", ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y


class Head(nn.Module):
    """Transformer block."""
    def __init__(self, head_size):
        super().__init__()
        self.wQ, self.wK, self.wV = self.initialize_weights(n_embed)

        # The head_size is the dimension after the linear transformation
        # It does not need to be the same as n_embed, it just needs to be consistent
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
    
    def forward(self, X):
        """Take in a 2 or 3d tensor and calculate output embeddings.
        
        :param X: pytorch tensor with dims (B, T, C) or (T, C)
        """

        B, T, C = X.shape

        # Create the query, key, and value matrices
        # Weights are CxC - the embedding dim dimension
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # Take the dot product with each previous matrix
        # Q is of shape (B, T, C) and KT is of shape (B, C, T)
        # Pytorch broadcasting does the the TxC dot CxT matrix multiplication for each batch in dim B
        xdot = (Q @ torch.transpose(K, -2, -1)).masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Normalize by the square root of the input dim
        xdot = xdot/math.sqrt(X.shape[-1])
        
        # Softmax to get weights of each previous element 
        alpha = F.softmax(xdot, dim=1)
        
        # Multiply by X again to get a matrix with Y (each row is dim C)
        Y = alpha @ V
        return Y

    def initialize_weights(self, C):
        """Create the traininable weight matrices for the query, key and value projections."""
        wQ = torch.rand(C, C, requires_grad=True)
        wK = torch.rand(C, C, requires_grad=True)
        wV = torch.rand(C, C, requires_grad=True)

        return wQ, wK, wV


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run all of the heads in parallel and concatenate the results over the C dimension
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """A simple feed forward layer with non-linearity"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            # This is very simple right now - just a single linear layer
            # It is effectively using the embedding output
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # For now we're using n_embed for the head_size, but it can project down to a smaller dim
        # For multi head, we split the original embedding into different channels, and use them independently
        # Because we are dividing by the number of heads, the concantenated output will have the same size as the input
        self.multi_attention_block = MultiHeadAttention(4, n_embed//4) # 4 heads, each with n_embed/4 size
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """ Forward pass. 
        
        This is an extremely simple model currently, where the embedding is used
        directly as an input into softmax to create an array of probabilities.  So the 
        embedding dimension must be equal to the vocab size since the emedding values itself
        are just the predictions.  So the model is taking each character and determining what
        the most likely next character is, trained on the offset by one x and y values."""
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape

        # Add a linear layer to go from token embeddings to logits now
        # This takes in an existing (B, T) shape set of indicies and gets their embeddings
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) - (Batch (4), Time (8), Channel(n_embed))
        position_embeddings = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        # Add position and token embeddings together (broadcasted over batches)
        x = token_embeddings + position_embeddings # (B,T,C)
        # Run through the transformer block
        x = self.multi_attention_block(x) # (B,T,C)
        # Feed forward layer that is applied individually for each token - so a linear transformation of the 
        # output embedding
        x = self.ffwd(x) # (B,T,C)
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

    lambda1 = lambda epoch: lr_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, scheduler


def train(model, train_data, val_data):
    """Train the model on the training data and evaluate on the validation data.
    
    Training gets a batch consisting of random starting points and the block size.
    The model uses a fixed set of iterations of generations per epoch and caclulates the
    train and validation loss at the end of each epoch."""


    # Set up the model and optimizer
    optimizer, scheduler = get_optimization_details(model)

    # Iterate through epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        model.train()

        # initialize tqdm object
        progress_bar = tqdm(total=train_num_batches, desc=f"Epoch {epoch+1}/{num_epochs}")

        batch_losses = []
        
        # Iterate through batches
        for _ in range(train_num_batches):
            xb, yb = get_batch("train", train_data, val_data)
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

        # Calculate the validation loss once per epoch

        with torch.no_grad():
            model.eval()
            val_batch_losses = []
            # Iterate through batches
            for _ in range(val_num_batches):
                xb, yb = get_batch("val", train_data, val_data)
                xb = xb.to(device)
                yb = yb.to(device)
                # forward pass
                logits, loss = model(xb, yb)

                val_batch_losses.append(loss.item())

            val_total_loss = sum(val_batch_losses)/len(val_batch_losses)

              
        # Print out training loop stats
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_total_loss:.4f}, Execution time: {epoch_time:.4f}")
        progress_bar.close()

def main():
    """ Main entry point of the app."""

    data, vocab_size, stoi, itos = tokenize_data("input.txt")

    print(data[:100])

    model = BigramLanguageModel(vocab_size).to(device)

    train_data, val_data = train_test_split(data, val_pct)
    print("Train/Val data len: ", len(train_data), len(val_data))

    train(model, train_data, val_data)

    # Use the (currently untrained) model to generate new characters
    idx = torch.tensor([[51, 39, 62, 0]]).to(device)
    print(decode(model.generate(idx, 200)[0].tolist(), itos))

if __name__ == "__main__":
    """ This is executed when run from the command line."""
    main()
