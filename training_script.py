"""Train the transformer model."""


import torch  
import torch.optim as optim
import time
from tqdm import tqdm
import json

from Transformer import (
    TransformerLanguageModel,
    encode,
    decode,
    tokenize_data,
)

def train_test_split(data, val_pct):
    """Train test split on the data."""

    n = int((1-val_pct)*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(split, train_data, val_data, config): # train or validation split
    """Generate a small batch of data from inputs x and targets y."""
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be train or val")
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],)) # batch_size random sequence starting points
    # print("Random starting points for each block: ", ix)
    x = torch.stack([data[i:i+config["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+1+config["block_size"]] for i in ix])
    return x, y


def get_optimization_details(model, config):

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    lambda1 = lambda epoch: config["lr_decay"] ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, scheduler


def train(model, train_data, val_data, config, device, train_num_batches, val_num_batches):
    """Train the model on the training data and evaluate on the validation data.
    
    Training gets a batch consisting of random starting points and the block size.
    The model uses a fixed set of iterations of generations per epoch and caclulates the
    train and validation loss at the end of each epoch."""


    # Set up the model and optimizer
    optimizer, scheduler = get_optimization_details(model, config)
    num_epochs = config["num_epochs"]

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
            xb, yb = get_batch("train", train_data, val_data, config)
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
                xb, yb = get_batch("val", train_data, val_data, config)
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


def set_parameters():

    # Load the config file json
    f = open("training_config.json")
    config = json.load(f)
    torch.manual_seed(1337)

    train_num_batches = config["train_num_steps"] // config["batch_size"]
    val_num_batches = config["val_num_steps"] // config["batch_size"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return config, device, train_num_batches, val_num_batches


def main():
    """ Main entry point of the app."""

    config, device, train_num_batches, val_num_batches = set_parameters()
    data, vocab_size, stoi, itos = tokenize_data("input.txt")
    print(data[:100])


    model = TransformerLanguageModel(vocab_size, config, device).to(device)
    train_data, val_data = train_test_split(data, config["val_pct"])
    print("Train/Val data len: ", len(train_data), len(val_data))

    train(model, train_data, val_data, config, device, train_num_batches, val_num_batches)

    # Use the trained model to generate new characters
    idx = torch.tensor([[51, 39, 62, 0]]).to(device)
    print(decode(model.generate(idx, 500)[0].tolist(), itos))

if __name__ == "__main__":
    """ This is executed when run from the command line."""
    main()