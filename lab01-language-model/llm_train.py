# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import os

import matplotlib.pyplot as plt
import mlflow
import requests
import torch
from torchinfo import summary
from transformers import AutoTokenizer

from dataset import create_dataloader_v1, create_speakleash_dataloader
from logger import get_configured_logger
from models import DataLoaderConfig, ModelTrainingConfig, TransformerModelConfig

# Import from local files
from transformer_based_llm import GPTModel, generate_text_simple

########################
# MODEL MONITORING
############
logger = get_configured_logger("gpt_train", log_file="logs/gpt_train.log")
remote_server_uri = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("/language-model-transformer")
########################
# MODEL MONITORING
############


##################
# GLOBAL VARIABLES
##################
optimizer_mapper = {
    "adamw": torch.optim.AdamW,
}
########################
# GLOBAL VARIABLES
############


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
    train_perplexity = torch.exp(torch.tensor(train_loss))
    val_perplexity = torch.exp(torch.tensor(val_loss))
    return train_loss, val_loss, train_perplexity, val_perplexity


def generate_and_print_sample(model: GPTModel, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size, use_sampling=True
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        logger.info(f"Sample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")
        mlflow.log_text(decoded_text, "artifacts/generated_text.txt")
    model.train()


def train_model_simple(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    max_new_tokens=50,
):
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
                train_loss, val_loss, train_perplexity, val_perplexity = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(
                    f"Epoch: {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Train perplexity {train_perplexity:.3f}, Val perplexity {val_perplexity:.3f}"
                )
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_perplexity": train_perplexity,
                        "val_perplexity": val_perplexity,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
                # Print a sample text after each epoch
                generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=max_new_tokens)

    return train_losses, val_losses, track_tokens_seen


def evaluate_test_model(model: GPTModel, test_loader, device, tokenizer, start_context="Every effort moves you", max_new_tokens=100):
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
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size, use_sampling=True
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Generate sample text
    logger.info(f"\nSample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")

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


def main(model_config, training_config, dataset_config, tokenizer):
    torch.manual_seed(123)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(model_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    ##############################
    #  Model summary
    ##############################
    dummy_input = torch.randint(0, model_config.get("vocab_size"), (1, 256), dtype=torch.long).to(device)
    model_summary = str(summary(model, input_data=dummy_input))
    mlflow.log_text(model_summary, "artifacts/model_summary.txt")

    ##############################
    # Initialize optimizer and tokenizer
    ##############################
    optimizer = optimizer_mapper[training_config["optimizer"]](
        model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"]
    )
    max_docs = dataset_config.get("max_docs", None)
    tokenizer_name = tokenizer.name if hasattr(tokenizer, "name") else tokenizer.__class__.__name__
    mlflow.log_params(
        {
            "device": str(device),
            "tokenizer": tokenizer_name,
            "optimizer_details": optimizer.__str__(),
        }
    )

    #########################ń#####
    # Set up dataloaders
    ##############################
    # Speaklesh dataset
    if dataset_config.get("use_speaklesh", False):
        logger.info("Using Speakleash dataset for training/validation/testing")
        # Use smaller stride for better overlap and more training data
        stride = model_config["context_length"] // 2  # Use 1/4 of context length for better overlap
        train_loader = create_speakleash_dataloader(
            split="train",
            batch_size=training_config["batch_size"],
            max_docs=max_docs,
            stride=stride,
            tokenizer=tokenizer,
            max_length=model_config["context_length"],
            speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
        )
        val_loader = create_speakleash_dataloader(
            split="val",
            batch_size=training_config["batch_size"],
            max_docs=max_docs,
            stride=stride,
            tokenizer=tokenizer,
            max_length=model_config["context_length"],
            speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
        )
        test_loader = create_speakleash_dataloader(
            split="test",
            batch_size=training_config["batch_size"],
            max_docs=max_docs,
            stride=stride,
            tokenizer=tokenizer,
            max_length=model_config["context_length"],
            speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
        )

    else:
        logger.info("Using custom dataset for training/validation/testing")
        # The Verdict dataset from tutorial

        # Train/validation ratio
        ##############################
        # Download data if necessary
        ##############################

        file_path = "the-verdict.txt"
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

        if not os.path.exists(file_path):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_data = response.text
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            train_ratio = 0.80
            split_idx_train = int(train_ratio * len(text_data))
            split_idx_val = int((train_ratio + 0.10) * len(text_data))
            # Use smaller stride for better overlap and more training data
            stride = model_config["context_length"] // 4
            train_loader = create_dataloader_v1(
                text_data[:split_idx_train],
                batch_size=training_config["batch_size"],
                max_length=model_config["context_length"],
                stride=stride,
                drop_last=True,
                shuffle=True,
                num_workers=0,
            )

            val_loader = create_dataloader_v1(
                text_data[split_idx_train:split_idx_val],
                batch_size=training_config["batch_size"],
                max_length=model_config["context_length"],
                stride=stride,
                drop_last=False,
                shuffle=False,
                num_workers=0,
            )

            test_loader = create_dataloader_v1(
                text_data[split_idx_val:],
                batch_size=training_config["batch_size"],
                max_length=model_config["context_length"],
                stride=stride,
                drop_last=False,
                shuffle=False,
                num_workers=0,
            )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=training_config["num_epochs"],
        eval_freq=training_config["eval_freq"],
        eval_iter=training_config["eval_iter"],
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
        max_new_tokens=model_config.get("max_new_tokens"),
    )

    return train_losses, val_losses, tokens_seen, model, test_loader


if __name__ == "__main__":
    ################################
    ########### TOKENIZER ###########
    ################################
    tokenizer = AutoTokenizer.from_pretrained("flax-community/papuGaPT2")
    # tokenizer= tiktoken.get_encoding("gpt2")
    ################################
    ########### TOKENIZER ###########
    ################################

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    TRANSFORMER_MODEL_CONFIG = TransformerModelConfig(
        **{
            "vocab_size": vocab_size,  # Vocabulary size
            "context_length": 256,  # Shortened context length (orig: 1024)
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.35,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
            "max_new_tokens": 256,
        }
    ).model_dump()

    TRAINING_SETTINGS = ModelTrainingConfig(
        **{
            "learning_rate": 45e-4,
            "num_epochs": 20,
            "batch_size": 4,
            "weight_decay": 0.5,
            "optimizer": "adamw",
            "eval_freq": 20,
            "eval_iter": 5,
        }
    ).model_dump()

    DATASET_SETTINGS = DataLoaderConfig(
        **{"max_docs": 200, "use_speaklesh": True, "speaklesh_dataset_name": "wolne_lektury_corpus"}
    ).model_dump()

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################

    ###########################
    # Initiate training
    ###########################
    START_CONTEXT = "Pierogi ruskie to"
    with mlflow.start_run():
        mlflow.log_params(TRANSFORMER_MODEL_CONFIG)
        mlflow.log_params(TRAINING_SETTINGS)
        mlflow.log_params(DATASET_SETTINGS)

        train_losses, val_losses, tokens_seen, model, test_loader = main(
            TRANSFORMER_MODEL_CONFIG, TRAINING_SETTINGS, DATASET_SETTINGS, tokenizer=tokenizer
        )

        ###########################
        # After training
        ###########################

        # Evaluate model on test data

        test_results = evaluate_test_model(
            model,
            test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
            tokenizer=tokenizer,
            start_context=START_CONTEXT,
            max_new_tokens=TRANSFORMER_MODEL_CONFIG.get("max_new_tokens"),
        )
        mlflow.log_metrics({"test_loss": test_results["test_loss"]})

        # Save and load model
        torch.save(model.state_dict(), f"models/model_id_{mlflow.active_run().info.run_id}.pth")
        model = GPTModel(TRANSFORMER_MODEL_CONFIG)
        model.load_state_dict(torch.load(f"models/model_id_{mlflow.active_run().info.run_id}.pth", weights_only=True))

        logger.info("\nFinal Test Results:")
        logger.info(f"Test Loss: {test_results['test_loss']:.4f}")
