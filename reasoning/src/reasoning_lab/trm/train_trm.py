import argparse
import time

import torch
from tiny_recursive_model import MLPMixer1D, TinyRecursiveModel, Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from reasoning_lab.dataset.sudoku_dataset import SudokuDataset
from reasoning_lab.trm.logger import get_configured_logger
from reasoning_lab.utils import format_sudoku_grid, is_empty, range_from_one  # Also applies TRM predict fix

# Initialize logger
logger = get_configured_logger("sudoku_train", log_file="logs/train.log")


def evaluate_accuracy(model, dataloader, device, num_cnt=None, log_prefix="eval", step=None):
    # Handle DataParallel / DistributedDataParallel wrapper
    if hasattr(model, "module"):
        model = model.module

    # Handle MPS issue in trm.predict
    eval_device = device
    if device == "mps":
        eval_device = "cpu"

    # Force model to eval_device to ensure Embeddings are on correct device
    model.to(eval_device)

    model.eval()
    correct = 0
    total = 0

    # Evaluate on a subset or full dataset
    # Try to get dataset length from dataloader if possible, else just rely on loop/num_cnt
    dataset_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else float("inf")
    limit = min(dataset_len, num_cnt) if num_cnt is not None else dataset_len

    logger.info(f"Evaluating on {limit} examples on {eval_device}...")

    correct_examples = []
    incorrect_examples = []

    pbar = tqdm(total=limit if limit != float("inf") else None)

    for inputs, targets in dataloader:
        if total >= limit:
            break

        # Adjust batch if execution exceeds limit (shouldn't happen often)
        current_batch_size = inputs.size(0)
        if total + current_batch_size > limit:
            inputs = inputs[: limit - total]
            targets = targets[: limit - total]
            current_batch_size = inputs.size(0)

        inputs = inputs.to(eval_device)
        targets = targets.to(eval_device)

        with torch.no_grad():
            # Use appropriate inference params
            # model.predict handles batches? Assuming YES based on tensor inputs
            # If not, we iterate
            try:
                pred_answers, _ = model.predict(inputs, max_deep_refinement_steps=16, halt_prob_thres=0.1)
            except RuntimeError:
                # Fallback to loop if Predict doesn't handle batching for some reason (unlikely for TRM)
                pred_answers = []
                for i in range(inputs.size(0)):
                    pa, _ = model.predict(inputs[i : i + 1], max_deep_refinement_steps=16, halt_prob_thres=0.1)
                    pred_answers.append(pa)
                pred_answers = torch.cat(pred_answers, dim=0)

        # Compare
        # pred_answers: [B, 81]
        matches = torch.all(pred_answers == targets, dim=1)  # [B]
        correct += matches.sum().item()
        total += current_batch_size
        pbar.update(current_batch_size)

        # Collect examples
        if len(correct_examples) < 3 or len(incorrect_examples) < 3:
            inputs_cpu = inputs.cpu()
            targets_cpu = targets.cpu()
            preds_cpu = pred_answers.cpu()

            for i in range(current_batch_size):
                is_correct = matches[i].item()
                if is_correct and len(correct_examples) < 3:
                    correct_examples.append((inputs_cpu[i], preds_cpu[i], targets_cpu[i]))
                elif not is_correct and len(incorrect_examples) < 3:
                    incorrect_examples.append((inputs_cpu[i], preds_cpu[i], targets_cpu[i]))

    pbar.close()

    # Restore model device if changed
    if device == "mps":
        model.to(device)

    model.train()  # Switch back to train mode

    acc = correct / total if total > 0 else 0.0
    num_incorrect = total - correct

    # Log examples to W&B
    if wandb.run is not None:
        columns = ["Status", "Input", "Predicted", "Target"]
        data = []
        for inp, pred, tgt in correct_examples:
            data.append(["Correct", format_sudoku_grid(inp), format_sudoku_grid(pred), format_sudoku_grid(tgt)])
        for inp, pred, tgt in incorrect_examples:
            data.append(["Incorrect", format_sudoku_grid(inp), format_sudoku_grid(pred), format_sudoku_grid(tgt)])

        # Include step in table key to prevent overwriting
        table_key = f"{log_prefix}/examples_step_{step}" if step is not None else f"{log_prefix}/examples"
        wandb.log(
            {
                table_key: wandb.Table(columns=columns, data=data),
                f"{log_prefix}/num_correct": correct,
                f"{log_prefix}/num_incorrect": num_incorrect,
                f"{log_prefix}/accuracy": acc,
            },
            step=step,
        )
        logger.info(f"Logged {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples to W&B ({log_prefix}).")

    return acc, correct, num_incorrect


class WandBTrainer(Trainer):
    def __init__(self, *args, val_dataset=None, max_time_minutes=None, checkpoint_path="best_model.pt", log_every=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_dataset = val_dataset
        self.max_time_minutes = max_time_minutes
        self.checkpoint_path = checkpoint_path
        self.start_time = time.time()
        self.best_val_acc = 0.0
        self.log_every = log_every
        self.global_step = 0

    def evaluate_validation(self, epoch):
        if not self.val_dataset:
            return

        logger.info(f"Starting validation for epoch {epoch}, global step {self.global_step}")

        # 1. Calculate Validation Loss (similar to training step)
        total_val_loss = 0.0
        total_main_loss = 0.0
        total_halt_loss = 0.0
        batches = 0

        # Use a simplified loader for loss calculation
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for dataset_input, dataset_output in val_loader:
                dataset_input = dataset_input.to(self.accelerator.device)
                dataset_output = dataset_output.to(self.accelerator.device)

                outputs, latents = self.model.get_initial()

                loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(
                    dataset_input, outputs, latents, labels=dataset_output
                )

                total_val_loss += loss.item()
                total_main_loss += main_loss.mean().item()
                total_halt_loss += halt_loss.mean().item()
                batches += 1

        avg_val_loss = total_val_loss / batches if batches > 0 else 0
        avg_main_loss = total_main_loss / batches if batches > 0 else 0
        avg_halt_loss = total_halt_loss / batches if batches > 0 else 0

        # 2. Calculate Validation Accuracy
        val_acc_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=True)
        acc, val_correct, val_incorrect = evaluate_accuracy(
            self.model, val_acc_loader, self.accelerator.device, num_cnt=100, log_prefix="val", step=self.global_step
        )

        logger.info(
            f"Epoch {epoch} Validation: Acc {acc * 100:.2f}% | Loss {avg_val_loss:.4f} (Main: {avg_main_loss:.4f}, Halt: {avg_halt_loss:.4f})"
        )

        wandb.log(
            {
                "val/accuracy": acc,
                "val/total_loss": avg_val_loss,
                "val/main_loss": avg_main_loss,
                "val/halt_loss": avg_halt_loss,
                "epoch": epoch,
            },
            step=self.global_step,
        )

        # Checkpointing
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), self.checkpoint_path)
            logger.info(f"Saved new best model to {self.checkpoint_path}")

        self.model.train()

    def forward(self):
        for epoch in range_from_one(self.epochs):
            for dataset_input, dataset_output in self.dataloader:
                # Check time limit
                elapsed_min = (time.time() - self.start_time) / 60
                if self.max_time_minutes and elapsed_min > self.max_time_minutes:
                    if self.accelerator.is_main_process:
                        logger.info(f"Time limit of {self.max_time_minutes} minutes reached. Stopping training.")
                        # wandb.finish() - letting main() handle this
                    return  # Exit training

                outputs, latents = self.model.get_initial()

                self.global_step += 1

                total_loss_sum = 0.0
                main_loss_sum = 0.0
                halt_loss_sum = 0.0
                steps_count = 0

                for recurrent_step in range_from_one(self.max_recurrent_steps):
                    loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(
                        dataset_input, outputs, latents, labels=dataset_output
                    )

                    # Accumulate losses
                    total_loss_sum += loss.item()
                    main_loss_sum += main_loss.mean().item()
                    halt_loss_sum += halt_loss.mean().item()
                    steps_count += 1

                    self.accelerator.backward(loss)

                    self.optim.step()
                    self.optim.zero_grad()

                    self.scheduler.step()

                    if self.accelerator.is_main_process:
                        self.ema_model.update()

                    # handle halting
                    halt_mask = halt >= self.halt_prob_thres

                    if not halt_mask.any():
                        pass  # Continue accumulation logic, but check exit below
                    else:
                        outputs = outputs[~halt_mask]
                        latents = latents[~halt_mask]
                        dataset_input = dataset_input[~halt_mask]
                        dataset_output = dataset_output[~halt_mask]

                    if is_empty(outputs):
                        break

                # Logging (once per batch, averaged over recurrent steps)
                if self.global_step % self.log_every == 0:
                    avg_total = total_loss_sum / steps_count if steps_count > 0 else 0
                    avg_main = main_loss_sum / steps_count if steps_count > 0 else 0
                    avg_halt = halt_loss_sum / steps_count if steps_count > 0 else 0

                    if self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "train/total_loss": avg_total,
                                "train/main_loss": avg_main,
                                "train/halt_loss": avg_halt,
                                "train/epoch": epoch,
                            },
                            step=self.global_step,
                        )

                    if self.accelerator.is_main_process:
                        logger.info(f"[{epoch} (step {self.global_step})] loss: {avg_main:.3f} | halt: {avg_halt:.3f}")

                    # Periodic Validation
                    if self.val_dataset and self.accelerator.is_main_process:
                        self.evaluate_validation(epoch)

            # Wait for main process if needed? accelerate handles most.

        if self.accelerator.is_main_process:
            logger.info("complete")

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()


def main():
    parser = argparse.ArgumentParser(description="Train TRM on Sudoku")
    parser.add_argument("--max_training_time", type=int, default=240, help="Maximum training time in minutes (default: 240)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    args = parser.parse_args()

    # Load environment variables
    import os

    from dotenv import load_dotenv

    load_dotenv()

    # --- Configuration ---
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DIM = 512
    DEPTH = 4
    NUM_TOKENS = 10
    SEQ_LEN = 81
    MAX_TIME_MIN = args.max_training_time

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    logger.info(f"Using device: {DEVICE}")

    # Initialize W&B
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "dim": DIM,
            "depth": DEPTH,
            "device": DEVICE,
            "max_time_minutes": MAX_TIME_MIN,
            "expansion_factor": 4,
            "num_refinement_blocks": 3,
            "num_latent_refinements": 6,
        },
    )

    # Create run directory
    run_dir = f"runs/{run.id}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory created: {run_dir}")

    # --- Define W&B Metrics for Custom X-Axis ---
    # Removed as per user request (using explicit step argument instead)

    # --- Data ---
    # 100k train, 1k val from train split
    train_ds = SudokuDataset(split="train", start_idx=0, end_idx=100000, augment=False)
    val_ds = SudokuDataset(split="train", start_idx=100000, end_idx=101000, augment=False)

    logger.info(f"Training on ~{len(train_ds)} examples for max {MAX_TIME_MIN} mins")

    # --- Model ---
    trm = TinyRecursiveModel(
        dim=DIM,
        num_tokens=NUM_TOKENS,
        network=MLPMixer1D(dim=DIM, depth=DEPTH, seq_len=SEQ_LEN, expansion_factor=4),
        num_refinement_blocks=3,  # H_cycles - T in paper
        num_latent_refinements=6,  # L_cycles - N in paper
    ).to(DEVICE)

    use_cpu = DEVICE == "cpu"

    trainer = WandBTrainer(
        trm,
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        cpu=use_cpu,
        val_dataset=val_ds,  # Pass validation dataset
        max_time_minutes=MAX_TIME_MIN,
        checkpoint_path=f"{run_dir}/best_model.pt",
    )

    logger.info("Starting training...")
    trainer()

    # Save final model as well
    torch.save(trm.state_dict(), f"{run_dir}/final_model.pt")

    # --- Test Set Evaluation ---
    logger.info("\nStarting Test Set Evaluation...")

    # Load test set
    test_ds = SudokuDataset(split="test", augment=False, start_idx=0, end_idx=1000)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate
    # Evaluating on full test set might take time, use CPU safe evaluation
    acc, correct, incorrect = evaluate_accuracy(trm, test_loader, DEVICE, num_cnt=None, log_prefix="test", step=trainer.global_step)
    logger.info(f"Test Set Accuracy: {acc * 100:.2f}% (Correct: {correct}, Incorrect: {incorrect})")
    wandb.log({"test_accuracy": acc}, step=trainer.global_step)

    wandb.finish()


if __name__ == "__main__":
    main()
