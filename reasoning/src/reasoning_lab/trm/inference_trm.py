"""
Inference script for TinyRecursiveModel on Sudoku dataset.
Aligned with LLM evaluation format for easy comparison.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tiny_recursive_model import MLPMixer1D, TinyRecursiveModel

from reasoning_lab.llm.metrics import aggregate_metrics, calculate_exact_accuracy, calculate_partial_accuracy
from reasoning_lab.trm.logger import get_configured_logger

# Initialize logger
logger = get_configured_logger("sudoku_inference", log_file="logs/inference.log")

# Use same test sample IDs as LLM evaluation for consistency
TEST_SAMPLE_IDS = [0, 10, 20, 30, 40]


@dataclass
class EvaluationResult:
    """Single evaluation result - same structure as LLM evaluator."""

    llm_name: str  # Will be "trm" for TinyRecursiveModel
    strategy: str  # Will be "n/a" for TRM (no prompting strategy)
    example_id: int
    puzzle: list[int]
    expected: list[int]
    predicted: list[int] | None
    exact_match: bool
    partial_accuracy: float
    error: str | None = None


def load_test_examples() -> list[tuple[list[int], list[int], int]]:
    """
    Load fixed test examples from HuggingFace dataset.
    Same function as LLM evaluator for consistency.

    Returns:
        List of (puzzle, solution, example_id) tuples
    """
    ds = load_dataset("sapientinc/sudoku-extreme", split="test")
    examples = []

    for idx in TEST_SAMPLE_IDS:
        item = ds[idx]
        # Parse puzzle and solution strings
        q_str = item["question"].replace(".", "0")
        a_str = item["answer"].replace(".", "0")

        puzzle = [int(c) for c in q_str]
        solution = [int(c) for c in a_str]
        examples.append((puzzle, solution, idx))

    return examples


def run_inference_from_checkpoint(
    checkpoint_path: str,
    output_dir: str = "results/trm_evaluation",
    device: str = None,
    dim: int = 512,
    depth: int = 4,
    num_tokens: int = 10,
    seq_len: int = 81,
):
    """
    Load a checkpoint and perform inference on the same test examples as LLM evaluation.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file)
        output_dir: Directory to save results (same format as LLM evaluation)
        device: Device to run inference on (auto-detected if None)
        dim: Model dimension (must match trained model)
        depth: Network depth (must match trained model)
        num_tokens: Number of tokens/vocabulary size
        seq_len: Sequence length (81 for Sudoku)

    Returns:
        dict: Results in same format as LLM evaluation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Running inference on device: {device}")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Recreate model architecture (must match training config)
    model = TinyRecursiveModel(
        dim=dim,
        num_tokens=num_tokens,
        network=MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len, expansion_factor=4),
        num_refinement_blocks=3,
        num_latent_refinements=6,
    )

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Load test examples (same as LLM evaluation)
    print("\nLoading test examples...")
    examples = load_test_examples()
    print(f"Loaded {len(examples)} examples with IDs: {[e[2] for e in examples]}")

    # Handle MPS device issue (use CPU for inference if MPS)
    eval_device = "cpu" if device == "mps" else device
    if eval_device != device:
        model.to(eval_device)

    results = []

    print(f"\n{'=' * 50}")
    print("Evaluating: TRM (TinyRecursiveModel)")
    print("=" * 50)

    with torch.no_grad():
        for puzzle, solution, example_id in examples:
            print(f"  Example {example_id}...", end=" ")

            try:
                # Convert puzzle to tensor
                puzzle_tensor = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0).to(eval_device)

                # Run inference
                pred_answers, exit_steps = model.predict(puzzle_tensor, max_deep_refinement_steps=16, halt_prob_thres=0.1)

                predicted = pred_answers[0].cpu().tolist()
                exit_step = exit_steps[0].item()

                # Calculate metrics
                exact_match = calculate_exact_accuracy(predicted, solution)
                partial_acc = calculate_partial_accuracy(predicted, solution)

                result = EvaluationResult(
                    llm_name="trm",
                    strategy="n/a",
                    example_id=example_id,
                    puzzle=puzzle,
                    expected=solution,
                    predicted=predicted,
                    exact_match=exact_match,
                    partial_accuracy=partial_acc,
                )

                print(f"exact={exact_match}, partial={partial_acc:.3f}, exit_step={exit_step}")

            except Exception as e:
                result = EvaluationResult(
                    llm_name="trm",
                    strategy="n/a",
                    example_id=example_id,
                    puzzle=puzzle,
                    expected=solution,
                    predicted=None,
                    exact_match=False,
                    partial_accuracy=0.0,
                    error=str(e),
                )
                print(f"ERROR: {e}")

            results.append(result)

    # Aggregate metrics
    print("\n" + "=" * 50)
    print("Aggregating results...")
    print("=" * 50)

    result_dicts = [asdict(r) for r in results]
    aggregated = aggregate_metrics(result_dicts)

    # Save results in same format as LLM evaluation
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "test_sample_ids": TEST_SAMPLE_IDS,
        "model": "TinyRecursiveModel",
        "results": result_dicts,
        "aggregated": {f"{k[0]}_{k[1]}": v for k, v in aggregated.items()},
    }

    # Save intermediate results (for consistency with LLM evaluator)
    intermediate_path = output_dir / "intermediate_trm.json"
    with open(intermediate_path, "w") as f:
        json.dump({"llm_name": "trm", "timestamp": datetime.now().isoformat(), "results": result_dicts}, f, indent=2)

    # Save final results
    final_path = output_dir / "final_results.json"
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to: {final_path}")

    # Print summary table
    print_summary_table(aggregated)

    return final_output


def print_summary_table(aggregated: dict) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Strategy':<12} {'Exact Acc':<12} {'Partial Acc':<12}")
    print("-" * 70)

    for key, metrics in sorted(aggregated.items()):
        if isinstance(key, tuple):
            model_name, strategy = key
        else:
            # Handle string keys from JSON
            parts = key.rsplit("_", 1)
            model_name = parts[0]
            strategy = parts[1] if len(parts) > 1 else "n/a"

        print(f"{model_name:<25} {strategy:<12} {metrics['exact_accuracy']:<12.3f} {metrics['partial_accuracy']:<12.3f}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on TRM Sudoku model (aligned with LLM evaluation)")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output-dir", type=str, default="results/trm_evaluation", help="Output directory for results")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")

    args = parser.parse_args()

    run_inference_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
    )
