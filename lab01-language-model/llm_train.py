# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
from typing import Literal
import time
import matplotlib.pyplot as plt
import requests
import torch
from dotenv import load_dotenv
from torchinfo import summary

import wandb
from dataset import create_dataloader_v1, create_speakleash_dataloader
from logger import get_configured_logger
from models import DataLoaderConfig, ModelTrainingConfig
from transformer_based_llm import GPTModel, generate_text_simple
from utils import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate_and_print_sample,
    get_model,
    get_model_config,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)

load_dotenv()

# MODEL MONITORING
logger = get_configured_logger("llm_train", log_file="logs/llm_train.log")
# wandb will be initialized in __main__
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
    run,  # Pass run object for wandb logging
    max_new_tokens=50,
    max_training_minutes=60,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    best_val_loss = float("inf")
    training_start = time.time()
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
                # Condition to stop training after max_training_minutes
                current_time = time.time()
                if (current_time - training_start) / 60 > max_training_minutes:
                    logger.info(f"Maximum training time of {max_training_minutes} minutes reached. Stopping training.")
                    return train_losses, val_losses, track_tokens_seen

                train_loss, val_loss, train_perplexity, val_perplexity = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(
                    f"Epoch: {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Train perplexity {train_perplexity:.3f}, Val perplexity {val_perplexity:.3f}"
                )
                run.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_perplexity": train_perplexity,
                        "val_perplexity": val_perplexity,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                # Print a sample text after each epoch
                generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=max_new_tokens)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save the best model
                    best_model_path = f"models/{run.id}/best_model_checkpoint.pth"
                    cur_start = time.time()
                    torch.save(model.state_dict(), best_model_path)
                    cur_end = time.time()
                    logger.info(f"Saved best model at step {global_step} to {best_model_path} (took {cur_end - cur_start:.2f} seconds)")

    return train_losses, val_losses, track_tokens_seen


def evaluate_test_model(model: GPTModel, test_loader, device, tokenizer, start_context="Every effort moves you", max_new_tokens=100, context_size=256):
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
        test_perplexity = torch.exp(torch.tensor(test_loss))

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Perplexity: {test_perplexity:.4f}")


    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size, use_sampling=True
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Generate sample text
    logger.info(f"\nSample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")

    return {"test_loss": test_loss, "sample_text": decoded_text, "test_perplexity": test_perplexity}


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


def main(
    model_config,
    training_config,
    dataset_config,
    tokenizer,
    run,
    model_type: Literal["rnn", "transformer"] = "transformer",
):
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

    model = get_model(model_type, model_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    ##############################
    #  Model summary
    ##############################
    dummy_input = torch.randint(0, model_config.get("vocab_size"), (1, 256), dtype=torch.long).to(device)
    model_summary = str(summary(model, input_data=dummy_input))
    run.log({"model_summary": wandb.Html(f"<pre>{model_summary}</pre>")})

    ##############################
    # Initialize optimizer and tokenizer
    ##############################
    optimizer = optimizer_mapper[training_config["optimizer"]](
        model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"]
    )
    max_docs = dataset_config.get("max_docs", None)
    tokenizer_name = tokenizer.name if hasattr(tokenizer, "name") else tokenizer.__class__.__name__
    run.config.update(
        {
            "device": str(device),
            "tokenizer_name": tokenizer_name,
            "optimizer_details": str(optimizer),
            **model_config,
            **training_config,
            **dataset_config,
            "model_type": model_type,
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
        run=run,  # Pass run object for wandb logging
        max_new_tokens=model_config.get("max_new_tokens"),
        max_training_minutes=training_config["max_training_minutes"],
    )

    return train_losses, val_losses, tokens_seen, model, test_loader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Run Transformer training")
    argparse.add_argument(
        "--model_type",
        choices=["rnn", "transformer"],
        default="transformer",
        help="Type of model to train: 'rnn' or 'transformer'",
    )
    argparse.add_argument(
        "--tokenizer",
        default="papuGaPT2",
        help="Tokenizer to use",
    )
    argparse.add_argument(
        "--use_tiktoken",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to use tiktoken for the tokenizer (true/false)",
    )
    argparse.add_argument("--dataset_name", type=str, default="wolne_lektury_corpus", help="Name of the Speakleash dataset to use")
    argparse.add_argument("--max_docs", type=int, default=200, help="Maximum number of documents to use from the dataset")
    argparse.add_argument("--max_training_minutes", type=int, default=60, help="Maximum training time in minutes")

    args = argparse.parse_args()

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config={
            "tokenizer_script_arg_name": args.tokenizer,
            "use_tiktoken": args.use_tiktoken,
        },
        settings=wandb.Settings(),
    )

    # create dir for saving model
    os.makedirs(f"models/{run.id}", exist_ok=True)
    ################################
    ########### TOKENIZER ###########
    ################################
    tokenizer = get_tokenizer(args.tokenizer, args.use_tiktoken)
    ################################
    ########### TOKENIZER ###########
    ################################

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size

    MODEL_CONFIG = get_model_config(args.model_type, vocab_size=vocab_size)

    TRAINING_SETTINGS = ModelTrainingConfig(
        **{
            "learning_rate": 5e-4,
            "num_epochs": 1000,
            "batch_size": 64,
            "weight_decay": 0.1,
            "optimizer": "adamw",
            "eval_freq": 50,
            "eval_iter": 10,
            "max_training_minutes": args.max_training_minutes,
        }
    ).model_dump()

    DATASET_SETTINGS = DataLoaderConfig(
        **{"max_docs": args.max_docs, "use_speaklesh": True, "speaklesh_dataset_name": args.dataset_name}
    ).model_dump()

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################

    ###########################
    # Initiate training
    ###########################
    START_CONTEXT = "Pierogi ruskie to to jedno z tradycyjnych dań kuchni polskiej"
    logger.info(f"wandb run id: {run.id}")
    logger.info(f"Started training for model type: {args.model_type} with tokenizer: {args.tokenizer}")
    # wandb.config is already updated in main()

    train_losses, val_losses, tokens_seen, model, test_loader = main(
        MODEL_CONFIG, TRAINING_SETTINGS, DATASET_SETTINGS, tokenizer=tokenizer, run=run, model_type=args.model_type
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
        max_new_tokens=MODEL_CONFIG.get("max_new_tokens"),
        context_length=MODEL_CONFIG.get("context_length"),
    )
    run.log({"test_loss": test_results["test_loss"], "test_perplexity": test_results["test_perplexity"]})

    # Save and load model
    model_path = f"models/{run.id}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    model = get_model(args.model_type, MODEL_CONFIG)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Model correctly saved and loaded back.")
    run.finish()
