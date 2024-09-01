"""Define the Transformer Model"""

import torch  
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import math


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


class Head(nn.Module):
    """Basic attention head."""
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.wQ, self.wK, self.wV = self.initialize_weights(n_embd)

        # The head_size is the dimension after the linear transformation
        # It does not need to be the same as n_embed, it just needs to be consistent
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)
    
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
        alpha = self.dropout(alpha)
        
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

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all of the heads in parallel and concatenate the results over the C dimension
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """A simple feed forward layer with non-linearity"""

    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # This is very simple right now - just a single linear layer
            # It is effectively using the embedding output
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Combine feed forward and attention layers in a single repeatable block."""
    def __init__(self, n_embd, block_size, config, n_head=4):
        super().__init__()
        # For multi head, we split the original embedding into different channels, and use them independently
        # Because we are dividing by the number of heads, the concantenated output will have the same size as the input
        head_size = n_embd // n_head
        self.multi_attention_block = MultiHeadAttention(n_head, head_size, n_embd, block_size, config["mha_dropout"]) # 4 heads, each with n_embd/4 size
        self.ffwd = FeedForward(n_embd, config["ffwd_dropout"])
        # Layer norm to be applied before self attention and ffwd
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        # Run through the multi attention step
        # Also add the original input to the output of the multi attention step for residual connection
        x = x + self.multi_attention_block(self.ln1(x)) # (B,T,C)

        # Feed forward layer that is applied individually for each token - so a linear transformation of the output embedding
        # Residual connections are implemented by adding the original input to the output of the feed forward layer
        x = x + self.ffwd(self.ln2(x)) # (B,T,C)
        return x


class TransformerLanguageModel(nn.Module):
    
    def __init__(self, vocab_size, config, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.vocab_size = vocab_size
        self.n_embd = config["n_embd"]
        self.block_size = config["block_size"]
        
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(self.n_embd, self.block_size, config),
            TransformerBlock(self.n_embd, self.block_size, config),
            TransformerBlock(self.n_embd, self.block_size, config),
            TransformerBlock(self.n_embd, self.block_size, config),
            nn.LayerNorm(self.n_embd)
        )
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """ Forward pass. 
        
        Input is an idx parameter of shape (B, T) for Batch, Time.
        This input is used to access the already tokenized character level indicies that are integers
        mapping each to a character. The forward pass then:
        - Uses an embedding layer to look up embeddings for each character index
        - Uses broadcasting to add the position embedding to each token embedding
        - Sends the position adjusted embeddings through the transformer blocks
        - Calculates the loss on the output of the transformer
        """
        # idx and targets are both (B, T) tensor of integers in the vocab
        B, T = idx.shape

        # Add a linear layer to go from token embeddings to logits now
        # This takes in an existing (B, T) shape set of vocab indicies and gets their embeddings
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C) - (Batch (4), Time (8), Channel(n_embed))
        position_embeddings = self.position_embedding_table(torch.arange(T, device = self.device)) # (T,C)
        position_adjusted_embeddings = token_embeddings + position_embeddings # (B,T,C) broadcasted
        # Pass the embeddings through the set of transformer blocks - the main part of this
        x = self.transformer_blocks(position_adjusted_embeddings) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        # Evaluate the loss (compare logits to the next character (targets))
        if targets == None:
            loss = None
        else: 
            B, T, V = logits.shape
            logits = logits.view(B * T, V)  # Stack the time pieces for each batch on top of each other batch
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens on top of the existing T tokens."""
        self.eval()
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
            if idx.shape[1] > self.block_size:
                idx = idx[:, -self.block_size:]
            idx_output = torch.cat((idx_output, idx_next), dim=1) # (B, T+1)
        self.train()
        return idx_output
