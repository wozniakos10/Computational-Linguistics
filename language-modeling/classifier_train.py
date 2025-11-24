# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
import random
import time
from typing import Literal

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchinfo import summary

import wandb
from dataset import create_all_huggingface_dataloader
from datasets import load_dataset
from logger import get_configured_logger
from models import DataLoaderConfig, ModelTrainingConfig
from transformer_based_llm import GPTModelSequenceClassifier
from utils import get_classifier_model, get_classifier_model_config, get_tokenizer

load_dotenv()

# MODEL MONITORING
logger = get_configured_logger("classifier_train", log_file="logs/classifier_train.log")
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


@torch.no_grad()
def evaluate_loader(
    model: GPTModelSequenceClassifier,
    data_loader: DataLoader,
    loader_split: Literal["train", "val", "test"],
    device: str | torch.device,
    tokenizer,
    max_iter: None | int = None,
    save_path: None | str = None,
):
    model.eval()
    losses = []
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    mcc_metric = evaluate.load("matthews_correlation")
    cm_metric = evaluate.load("confusion_matrix")

    results = {}

    start = time.time()
    for idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss.item())

        predictions = torch.argmax(logits, dim=-1)

        predictions_cpu = predictions.cpu()
        labels_cpu = labels.cpu()

        accuracy_metric.add_batch(predictions=predictions_cpu, references=labels_cpu)
        f1_metric.add_batch(predictions=predictions_cpu, references=labels_cpu)
        mcc_metric.add_batch(predictions=predictions_cpu, references=labels_cpu)
        cm_metric.add_batch(predictions=predictions_cpu, references=labels_cpu)

        if max_iter is not None and idx >= max_iter:
            logger.info(f"Finish evaluating for {loader_split} as reached max iteration {max_iter}")
            break

    # Calculate metrics
    results["loss"] = sum(losses) / len(losses)
    results["acc"] = accuracy_metric.compute()["accuracy"]
    results["f1_score_macro"] = f1_metric.compute(average="macro")["f1"]
    results["mcc"] = mcc_metric.compute()["matthews_correlation"]

    # save to path if exist
    if save_path is not None:
        # calculate confusion matrix
        confusion_mat = cm_metric.compute()["confusion_matrix"]
        # create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        disp.plot(cmap="Blues", ax=ax)
        plt.title(f"Confusion Matrix - {loader_split}")
        # save to file
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"confusion_matrix_{loader_split}.png"))
        # save to wandb
        wandb.log({f"{loader_split}/confusion_matrix": wandb.Image(fig)})
        logger.info(f"Saved confusion matrix to {save_path} and to wandb")
        plt.close(fig)

    if loader_split == "test":
        # log 5 examples for test loader
        batch_size = len(input_ids)
        sample_indices = random.sample(range(batch_size), min(5, batch_size))

        logger.info(f"\nSample Predictions for {loader_split} split (from last batch)")

        for i, sample_idx in enumerate(sample_indices, 1):
            text = tokenizer.decode(input_ids[sample_idx].cpu(), skip_special_tokens=True)
            true_label = labels_cpu[sample_idx].item()
            pred_label = predictions_cpu[sample_idx].item()

            logger.info(f"\nExample {i}:")
            logger.info(f"Text: {text}")
            logger.info(f"True Label: {true_label}")
            logger.info(f"Predicted Label: {pred_label}")
            logger.info(f"Correct: {'YES' if true_label == pred_label else 'NO'}")

    stop = time.time()
    logger.info(f"Evaluation for {loader_split} dataset took {stop - start:.3f} seconds")

    model.train()

    return results


def train_classifier_model(
    model: GPTModelSequenceClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    num_epochs: int,
    eval_freq: int,
    gradient_clip: float,
    run,  # Pass run object for wandb logging
    max_training_minutes: int = 60,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses = [], []

    global_step = -1
    best_val_loss = float("inf")
    training_start = time.time()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()  # Calculate loss gradients

            # clipping gradient for better stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()  # Update model weights using loss gradients
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                # Condition to stop training after max_training_minutes
                current_time = time.time()
                if (current_time - training_start) / 60 > max_training_minutes:
                    logger.info(f"Maximum training time of {max_training_minutes} minutes reached. Stopping training.")
                    return train_losses, val_losses

                train_eval = evaluate_loader(model, train_loader, "train", device, tokenizer, max_iter=20)
                val_eval = evaluate_loader(model, val_loader, "val", device, tokenizer, max_iter=20)

                train_losses.append(train_eval["loss"])
                val_losses.append(val_eval["loss"])
                logger.info(
                    f"Epoch: {epoch + 1} (Step {global_step:06d})\n"
                    f"  Loss:         Train={train_eval['loss']:.3f}, Val={val_eval['loss']:.3f}\n"
                    f"Accuracy: Train={train_eval['acc']}, Val={val_eval['acc']}\n"
                    f"F1-Score (macro): Train={train_eval['f1_score_macro']}, Val={val_eval['f1_score_macro']}\n"
                    f"Mathew Correlation: Train={train_eval['mcc']}, Val={val_eval['mcc']}"
                )
                run.log(
                    {
                        "train_loss": train_eval["loss"],
                        "val_loss": val_eval["loss"],
                        "train_accuracy": train_eval["acc"],
                        "val_accuracy": val_eval["acc"],
                        "train_f1_macro": train_eval["f1_score_macro"],
                        "val_f1_macro": val_eval["f1_score_macro"],
                        "train_mcc": train_eval["mcc"],
                        "val_mcc": val_eval["mcc"],
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

                if val_eval["loss"] < best_val_loss:
                    best_val_loss = val_eval["loss"]
                    # Save the best model
                    best_model_path = f"models/{run.id}/best_model_checkpoint.pth"
                    cur_start = time.time()
                    torch.save(model.state_dict(), best_model_path)
                    cur_end = time.time()
                    logger.info(f"Saved best model at step {global_step} to {best_model_path} (took {cur_end - cur_start:.2f} seconds)")
        scheduler.step()

    return train_losses, val_losses


def evaluate_classifier_model_test(model: GPTModelSequenceClassifier, test_loader: DataLoader, device: str | torch.device, tokenizer, save_path):
    """Evaluate the trained model on the test dataset.

    Args:
        model (GPTModelSequenceClassifier): _description_
        test_loader (DataLoader): _description_
        device (str | torch.device): _description_
        tokenizer (AutoTokenizer): _description_

    Returns:
        _type_: _description_
    """

    logger.info("EVALUATING MODEL ON TEST DATA")

    # Calculate test loss

    test_eval = evaluate_loader(
        model=model,
        data_loader=test_loader,
        loader_split="test",
        device=device,
        tokenizer=tokenizer,
        save_path=save_path,
    )

    logger.info(f"Test Loss: {test_eval['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_eval['acc']}")
    logger.info(f"Test F1-Score (macro): {test_eval['f1_score_macro']}")
    logger.info(f"Test Mathew Correlation: {test_eval['mcc']}")

    return test_eval


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
    model_type: Literal["custom_gpt2", "hugging_face"] = "custom_gpt2",
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

    model = get_classifier_model(model_type=model_type, model_config=model_config)
    model = model.to(device)
    ##############################
    #  Model summary
    ##############################

    ctx_len = model_config["context_length"]
    dummy_input = torch.randint(
        0,
        model_config["vocab_size"],
        (1, ctx_len),
        dtype=torch.long,
    ).to(device)
    model_summary = str(summary(model, input_data=dummy_input))
    run.log({"model_summary": wandb.Html(f"<pre>{model_summary}</pre>")})

    ##############################
    # Initialize optimizer and tokenizer
    ##############################
    optimizer = optimizer_mapper[training_config["optimizer"]](
        model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"]
    )
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

    train_loader, val_loader, test_loader = create_all_huggingface_dataloader(
        dataset_name=dataset_config["hugging_face_dataset_name"],
        tokenizer=tokenizer,
        max_length=model_config["context_length"],
        batch_size=training_config["batch_size"],
    )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses = train_classifier_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=training_config["num_epochs"],
        eval_freq=training_config["eval_freq"],
        gradient_clip=training_config["gradient_clip"],
        run=run,
        max_training_minutes=training_config["max_training_minutes"],
    )

    return train_losses, val_losses, model, test_loader


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
        choices=["custom_gpt2", "hugging_face"],
        default="custom_gpt2",
        help="Type of model to train: 'custom_gpt2' or 'hugging_face'",
    )
    argparse.add_argument(
        "--tokenizer",
        default="speakleash/Bielik-4.5B-v3",
        help="Tokenizer to use. Provide tokenizer name or path",
    )
    argparse.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["tiktoken", "transformers", "sentence_piece", "custom"],
        default="tiktoken",
        help="Type of tokenizer to use: tiktoken, transformers, sentence_piece, or custom",
    )
    argparse.add_argument(
        "--dataset_name",
        type=str,
        default="jziebura/polish_youth_slang_classification",
        help="Name of the Huggingface dataset to use",
    )
    argparse.add_argument("--max_training_minutes", type=int, default=60, help="Maximum training time in minutes")

    args = argparse.parse_args()

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        config={
            "tokenizer_script_arg_name": args.tokenizer,
            "tokenizer_type": args.tokenizer_type,
        },
        settings=wandb.Settings(),
    )

    # create dir for saving model
    os.makedirs(f"models/{run.id}", exist_ok=True)
    ################################
    ########### TOKENIZER ###########
    ################################
    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_type)
    ################################
    ########### TOKENIZER ###########
    ################################

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size

    # right now hardcoded for jziebura/polish_youth_slang_classification
    dataset_dict = load_dataset(args.dataset_name)
    label_number = len(np.unique(dataset_dict["train"]["sentyment"]))
    MODEL_CONFIG = get_classifier_model_config(args.model_type, tokenizer, label_number)

    TRAINING_SETTINGS = ModelTrainingConfig(
        **{
            "learning_rate": 1e-5,
            "num_epochs": 3,
            "batch_size": 32,
            "weight_decay": 0.35,
            "optimizer": "adamw",
            "eval_freq": 200,
            "max_training_minutes": args.max_training_minutes,
            "gradient_clip": 1,
        }
    ).model_dump()

    DATASET_SETTINGS = DataLoaderConfig(**{"use_speaklesh": False, "use_hugging_face": True, "hugging_face_dataset_name": args.dataset_name}).model_dump()

    ################################################
    ########### MODEL AND TRAINING CONFIG ###########
    ################################################

    ###########################
    # Initiate training
    ###########################
    logger.info(f"wandb run id: {run.id}")
    logger.info(f"Started training for model type: {args.model_type} with tokenizer: {args.tokenizer}")
    # wandb.config is already updated in main()

    train_losses, val_losses, model, test_loader = main(
        MODEL_CONFIG, TRAINING_SETTINGS, DATASET_SETTINGS, tokenizer=tokenizer, run=run, model_type=args.model_type
    )

    ###########################
    # After training
    ###########################
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Evaluate model on test data
    test_results = evaluate_classifier_model_test(
        model=model,
        test_loader=test_loader,
        device=device,
        tokenizer=tokenizer,
        save_path=f"models/{run.id}",
    )
    run.log(
        {
            "test_loss": test_results["loss"],
            "test_accuracy": test_results["acc"],
            "test_f1_score_macro": test_results["f1_score_macro"],
            "test_mcc": test_results["mcc"],
        }
    )

    # Save and load model
    model_path = f"models/{run.id}/final_model.pth"
    torch.save(model.state_dict(), model_path)

    model = get_classifier_model(args.model_type, MODEL_CONFIG)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model = model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Model correctly saved and loaded back.")
    run.finish()
