# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import mlflow

import torch
import tiktoken

# Import from local files
from transformer_based_llm import GPTModel, generate_text_simple
from dataset import create_plwiki_dataloader
from logger import get_configured_logger

logger = get_configured_logger("gpt_train", log_file="logs/gpt_train.log")
remote_server_uri = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("/language-model-transformer")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        logger.info(f"Sample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")
        mlflow.log_text(decoded_text, "artifacts/generated_text.txt")
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"Epoch: {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
                # Print a sample text after each epoch
                generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_test_model(model, test_loader, device, tokenizer, start_context="Every effort moves you"):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model: The trained GPT model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        tokenizer: Tokenizer for text generation
        start_context: Starting text for sample generation

    Returns:
        dict: Dictionary containing test loss and sample generation
    """

    logger.info("EVALUATING MODEL ON TEST DATA")

    # Calculate test loss
    model.eval()
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device)

    logger.info(f"Test Loss: {test_loss:.4f}")

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=100, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Generate sample text
    logger.info(f"\nSample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")

    # # Calculate perplexity (optional metric)
    # perplexity = torch.exp(torch.tensor(test_loss)).item()
    # perplexity_metric = Perplexity()
    # print(f"Test Perplexity: {perplexity:.4f}")

    model.train()  # Reset to training mode
    logger.info(f"Test loss: {test_loss:.4f}")
    return {"test_loss": test_loss, "sample_text": decoded_text}


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    torch.manual_seed(123)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    ##############################
    # Download data if necessary
    ##############################

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    ##############################
    # Set up dataloaders
    ##############################

    train_loader = create_plwiki_dataloader(
        split="train", batch_size=settings["batch_size"], max_docs=1000, stride=gpt_config["context_length"]
    )
    val_loader = create_plwiki_dataloader(
        split="val", batch_size=settings["batch_size"], max_docs=1000, stride=gpt_config["context_length"]
    )

    test_loader = create_plwiki_dataloader(
        split="test", batch_size=settings["batch_size"], max_docs=1000, stride=gpt_config["context_length"]
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=50,
        eval_iter=1,
        start_context="Pierogi ruskie to",
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model, test_loader


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    }

    OTHER_SETTINGS = {"learning_rate": 5e-4, "num_epochs": 10, "batch_size": 2, "weight_decay": 0.1}

    ###########################
    # Initiate training
    ###########################
    with mlflow.start_run():
        mlflow.log_params(GPT_CONFIG_124M)
        mlflow.log_params(OTHER_SETTINGS)
        train_losses, val_losses, tokens_seen, model, test_loader = main(GPT_CONFIG_124M, OTHER_SETTINGS)

        ###########################
        # After training
        ###########################

        # Plot results
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

        # Evaluate model on test data
        tokenizer = tiktoken.get_encoding("gpt2")
        test_results = evaluate_test_model(
            model,
            test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
            tokenizer=tokenizer,
            start_context="Every effort moves you",
        )
        mlflow.log_metrics({"test_loss": test_results["test_loss"]})

        # Save and load model
        torch.save(model.state_dict(), "model.pth")
        model = GPTModel(GPT_CONFIG_124M)
        model.load_state_dict(torch.load("model.pth", weights_only=True))

        print("\nFinal Test Results:")
        print(f"Test Loss: {test_results['test_loss']:.4f}")
