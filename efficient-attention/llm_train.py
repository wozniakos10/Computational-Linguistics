# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
import time
from typing import Dict, Literal

import numpy as np
import tiktoken
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer

import wandb
from dataset import create_speakleash_dataloader
from logger import get_configured_logger
from memory_tracker import MemoryTracker
from models import DataLoaderConfig, ModelTrainingConfig, TransformerModelConfig
from transformer_based_llm import GPTModel
from utils import (
    calc_loss_batch,
    calc_loss_loader,
    get_model,
    get_tokenizer,
)

load_dotenv()

# removing parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def adjust_batch_size_dynamicaly(
    model_config: Dict,
    dataset_config: Dict,
    training_config: Dict,
    device: torch.device,
    model_type: str,
    use_fp16: bool,
    tokenizer: tiktoken.Encoding | AutoTokenizer,
) -> int:
    current_batch_size = training_config["batch_size"]
    tested_batch_size_lst = [current_batch_size]

    while True:
        logger.info(f"Testing batch_size={current_batch_size}")

        # Clean memory before creating new model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # New model for each test
        model = get_model(model_type, model_config)
        model.to(device)

        optimizer = optimizer_mapper[training_config["optimizer"]](
            model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"]
        )

        try:
            data_loader = create_speakleash_dataloader(
                split="test",
                batch_size=current_batch_size,
                stride=model_config["context_length"],
                tokenizer=tokenizer,
                max_length=model_config["context_length"],
                speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
                num_workers=training_config["num_workers"],
            )

            # Test with 20 batches to ensure stability
            for idx, (input_batch, target_batch) in enumerate(data_loader):
                if idx >= 20:
                    if device.type == "cuda":
                        util = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(device).total_memory
                        logger.info(f"Memory utilization at batch_size={current_batch_size}: {util:.1%}")

                        if util > 0.85:
                            logger.warning(f"Memory utilization {util:.1%} too high, stopping here")
                            del model, optimizer, data_loader
                            torch.cuda.empty_cache()
                            return current_batch_size

                    logger.info(f"Successfully tested batch_size={current_batch_size}")

                    del model, optimizer, data_loader
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    current_batch_size += current_batch_size // 4  # increase by 25%
                    tested_batch_size_lst.append(current_batch_size)
                    break

                optimizer.zero_grad()
                # Simulate training step
                if use_fp16:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss, _ = calc_loss_batch(input_batch, target_batch, model, device, tokenizer, calculate_perplexity=False)

                else:
                    loss, _ = calc_loss_batch(input_batch, target_batch, model, device, tokenizer, calculate_perplexity=False)

                loss.backward()
                optimizer.step()

        except torch.OutOfMemoryError:
            logger.info(f"Out of memory at batch size {current_batch_size}")
            logger.info(f"Successfully tested {len(tested_batch_size_lst)} batch sizes.")

            # Clean memory
            try:
                del model, optimizer, data_loader
            except Exception:
                pass

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if len(tested_batch_size_lst) < 2:
                raise RuntimeError("Even smallest batch size doesn't fit!")

            safe_batch_size = tested_batch_size_lst[-2]
            logger.info(f"Last working batch size: {safe_batch_size}")

            return safe_batch_size

        except Exception as e:
            raise e


def train_model_simple(
    model: GPTModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding | AutoTokenizer,
    use_fp16: bool,
    run: wandb.Run,  # Pass run object for wandb logging
    generate_text_sample: bool = True,
    max_new_tokens=50,
    max_training_minutes=60,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    training_start = time.time()
    steps_time_lst = []
    N_STEPS_TIME_MEAN = 20
    run.config.update({"average_step_time_size": N_STEPS_TIME_MEAN})

    # Remove pytorch profiler as I couldn't track memory for record functions properly

    # prof_dir = f"./profiler_logs/{run.id}"
    # os.makedirs(prof_dir, exist_ok=True)
    # profiler_config = {"skip_first": 10, "wait": 30, "warmup": 1, "active": 4}
    # device_to_profile = device.type if device.type == "cuda" else "cpu"
    # sort_by_keyword = "self_" + device_to_profile + "_memory_usage"

    # def trace_handler(p):
    #     logger.info(f"\nProfiler step {p.step_num}")
    #     p.export_chrome_trace(f"{prof_dir}/trace_{p.step_num}.json")

    # # Determine which activities to profile based on available device
    # activities = [ProfilerActivity.CPU]
    # if device.type == "cuda":
    #     activities.append(ProfilerActivity.CUDA)
    #     logger.info("Profiling CPU + CUDA")
    # else:
    #     logger.info(f"Profiling CPU only (device: {device.type})")

    # # Scheduler: skip first N steps, then profile periodically
    # my_schedule = schedule(
    #     skip_first=profiler_config["skip_first"],
    #     wait=profiler_config["wait"],
    #     warmup=profiler_config["warmup"],
    #     active=profiler_config["active"],
    #     repeat=0,  # Repeat indefinitely (0 = infinite)
    # )

    # logger.info("Memory profiler ENABLED with scheduler:")
    # logger.info(f"  - Skip first {profiler_config['skip_first']} steps")
    # logger.info(f"  - Wait {profiler_config['wait']} steps between cycles")
    # logger.info(f"  - Warmup {profiler_config['warmup']} steps per cycle")
    # logger.info(f"  - Profile {profiler_config['active']} active steps per cycle")
    # logger.info(f"  - Traces saved to: {prof_dir}")
    # logger.info("  - Memory stats logged to wandb for ALL steps")

    # Initialize custom memory tracker
    memory_tracker = MemoryTracker(device)
    logger.info(f"Custom memory tracker initialized for device: {device}")

    #     with profile(
    #     activities=activities,
    #     schedule=my_schedule,
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False,
    #     on_trace_ready=trace_handler,
    # ) as prof:

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # Wrap entire training step

        for input_batch, target_batch in train_loader:
            global_step += 1
            start_step_time = time.time()
            memory_tracker.start_step()

            optimizer.zero_grad()
            if use_fp16:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss, perplexity = calc_loss_batch(
                        input_batch, target_batch, model, device, tokenizer, calculate_perplexity=False, memory_tracker=memory_tracker
                    )

            else:
                loss, perplexity = calc_loss_batch(
                    input_batch, target_batch, model, device, tokenizer, calculate_perplexity=False, memory_tracker=memory_tracker
                )

            memory_tracker.start_backward()
            loss.backward()
            memory_tracker.mark_backward()

            optimizer.step()

            end_step_time = time.time()
            step_duration = end_step_time - start_step_time
            steps_time_lst.append(step_duration)

            memory_tracker.mark_entire_step()

            # Log memory stats
            mem_stats = memory_tracker.get_stats()
            if mem_stats is not None:
                precision = "fp16" if use_fp16 else "fp32"
                run.log(
                    {
                        f"memory/{precision}/step_max_memory_gb": mem_stats["step_max_memory_gb"],
                        f"memory/{precision}/forward_gb": mem_stats["max_forward_memory_gb"],
                        f"memory/{precision}/backward_gb": mem_stats["max_backward_memory_gb"],
                    },
                    step=global_step,
                )

            if len(steps_time_lst) > N_STEPS_TIME_MEAN:
                last_n_steps_mean = np.mean(steps_time_lst)
                steps_time_lst = []
                run.log(
                    {
                        "average_step_time_seconds": last_n_steps_mean,
                    }
                )

            tokens_seen += input_batch.numel()

            # # Step profiler after each training step
            # prof.step()

            # Optional evaluation step - just for training without validation, to speed up
            if global_step % eval_freq == 0:
                if use_fp16:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        train_loss, perplexity = calc_loss_loader(train_loader, model, device, tokenizer, num_batches=eval_iter)
                else:
                    train_loss, perplexity = calc_loss_loader(train_loader, model, device, tokenizer, num_batches=eval_iter)
                train_losses.append(train_loss)
                track_tokens_seen.append(tokens_seen)

                logger.info(
                    f"Epoch: {epoch + 1} (Step {global_step:06d})\n  Loss: Train={train_loss:.3f}  Perplexity:   Train={perplexity:.3f}"
                )
                run.log(
                    {
                        "train_loss": train_loss,
                        "train_perplexity": perplexity,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            current_time = time.time()
            if (current_time - training_start) / 60 > max_training_minutes:
                run.log(
                    {
                        "entire_training_time_minutes": (current_time - training_start) / 60,
                    }
                )
                logger.info(f"Maximum training time of {max_training_minutes} minutes reached. Stopping training.")
                return train_losses, val_losses, track_tokens_seen

    current_time = time.time()
    run.log(
        {
            "entire_training_time_minutes": (current_time - training_start) / 60,
        }
    )
    return train_losses, val_losses, track_tokens_seen


def evaluate_test_model(
    model: GPTModel,
    test_loader: DataLoader,
    device: torch.device,
    tokenizer: tiktoken.Encoding | AutoTokenizer,
    model_config: Dict,
    training_config: Dict,
):
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
        if training_config["use_fp16"]:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                test_loss, test_perplexity = calc_loss_loader(test_loader, model, device, tokenizer)

        else:
            test_loss, test_perplexity = calc_loss_loader(test_loader, model, device, tokenizer)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Perplexity: {test_perplexity:.4f}")

    return {
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
    }


def main(
    model_config,
    device,
    training_config,
    dataset_config,
    tokenizer,
    run,
    model_type: Literal["rnn", "transformer"] = "transformer",
):
    ##############################
    # Initialize model
    ##############################

    model = get_model(model_type, model_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    # if training_config["use_gradient_checkpointing"]:
    #     model.gradient_checkpointing_enable()
    ##############################
    #  Model summary
    ##############################
    dummy_input = torch.randint(0, model_config.get("vocab_size"), (1, 256), dtype=torch.long).to(device)
    if training_config["use_fp16"]:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            model_summary = str(summary(model, input_data=dummy_input))
    else:
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
    # Speaklesh dataset
    if dataset_config["use_speaklesh"]:
        logger.info("Using Speakleash dataset for training/validation/testing")
        # As there is a lot of training data, stride can be equal to context length
        stride = model_config["context_length"]
        train_loader = create_speakleash_dataloader(
            split="train",
            batch_size=training_config["batch_size"],
            stride=stride,
            tokenizer=tokenizer,
            max_length=model_config["context_length"],
            speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
            num_workers=training_config["num_workers"],
        )

        test_loader = create_speakleash_dataloader(
            split="test",
            batch_size=training_config["batch_size"],
            stride=stride,
            tokenizer=tokenizer,
            max_length=model_config["context_length"],
            speakleash_dataset_name=dataset_config["speaklesh_dataset_name"],
            num_workers=training_config["num_workers"],
        )

    else:
        raise ValueError("Only speakleash dataset is supported now!")

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        optimizer,
        device,
        num_epochs=training_config["num_epochs"],
        eval_freq=training_config["eval_freq"],
        eval_iter=training_config["eval_iter"],
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
        use_fp16=training_config["use_fp16"],
        run=run,  # Pass run object for wandb logging,
        generate_text_sample=training_config["generate_text_sample"],
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
        help="Tokenizer to use. Provide tokenizer name or path",
    )
    argparse.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["tiktoken", "transformers"],
        default="tiktoken",
        help="Type of tokenizer to use: tiktoken, transformers",
    )
    argparse.add_argument("--dataset_name", type=str, default="wolne_lektury_corpus", help="Name of the Speakleash dataset to use")
    argparse.add_argument("--max_training_minutes", type=int, default=60, help="Maximum training time in minutes")
    argparse.add_argument("--adjust_batch_size", type=str2bool, default=False, help="Whether to adjust batch size")
    argparse.add_argument("--use_flash_attention", type=str2bool, default=False, help="Whether to use flash attention for training")
    argparse.add_argument(
        "--use_gradient_checkpointing", type=str2bool, default=False, help="Whether to use gradient checkpointing for training"
    )
    argparse.add_argument("--use_fp16", type=str2bool, default=False, help="Whether to use fp16 (bfloat16) for training")
    args = argparse.parse_args()

    torch.manual_seed(123)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

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
    MODEL_CONFIG = TransformerModelConfig(
        vocab_size=vocab_size,
        context_length=256,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.4,
        qkv_bias=False,
        max_new_tokens=256,
        use_flash_attention=args.use_flash_attention,
        # default no window attention for (-1,-1)
        attention_window_size=(64, 0),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    ).model_dump()

    TRAINING_SETTINGS = ModelTrainingConfig(
        **{
            "learning_rate": 5e-4,
            "num_epochs": 1,
            "batch_size": 125,
            "weight_decay": 0.1,
            "optimizer": "adamw",
            "eval_freq": 100,
            "eval_iter": 1,
            "max_training_minutes": args.max_training_minutes,
            "generate_text_sample": False,
            "use_fp16": args.use_fp16,
            "num_workers": 0,
        }
    ).model_dump()

    DATASET_SETTINGS = DataLoaderConfig(**{"use_speaklesh": True, "speaklesh_dataset_name": args.dataset_name}).model_dump()

    if args.adjust_batch_size:
        logger.info("Adjusting batch size dynamically based on available memory...")

        adjusted_batch_size = adjust_batch_size_dynamicaly(
            MODEL_CONFIG,
            DATASET_SETTINGS,
            TRAINING_SETTINGS,
            device,
            model_type=args.model_type,
            use_fp16=TRAINING_SETTINGS["use_fp16"],
            tokenizer=tokenizer,
        )
        torch.cuda.empty_cache()
        TRAINING_SETTINGS["batch_size"] = adjusted_batch_size
        logger.info(f"Adjusted batch size to: {adjusted_batch_size}")

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
        MODEL_CONFIG, device, TRAINING_SETTINGS, DATASET_SETTINGS, tokenizer=tokenizer, run=run, model_type=args.model_type
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
        model_config=MODEL_CONFIG,
        training_config=TRAINING_SETTINGS,
    )
    run.log(
        {
            "test_loss": test_results["test_loss"],
            "test_perplexity": test_results["test_perplexity"],
        }
    )

    # Save and load model
    model_path = f"models/{run.id}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    model = get_model(args.model_type, MODEL_CONFIG)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Model correctly saved and loaded back.")
    run.finish()
