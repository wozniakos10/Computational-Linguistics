"""
LLM Evaluator for Sudoku Solving.

Evaluates multiple LLMs on sudoku solving with zero-shot and few-shot prompting,
using LangChain with structured output for consistent response parsing.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from reasoning_lab.llm.config import LLMApiConfig
from reasoning_lab.llm.metrics import calculate_exact_accuracy, calculate_partial_accuracy
from reasoning_lab.llm.models import SudokuGrid
from reasoning_lab.llm.prompts import SudokuPrompts

# Fixed test sample IDs for consistency across evaluations
TEST_SAMPLE_IDS = [0, 10, 20, 30, 40]

# LLM names to evaluate
# "gemini_2_5_flash"
# "gpt_oss_120b",
LLM_NAMES = [
    "gpt_oss_120b",
    "claude_opus_4_5",
    "gemini_2_5_flash",
]
# LLM_NAMES = ["claude_opus_4_5"]

# Prompt strategies
STRATEGIES = ["zero_shot", "few_shot"]


@dataclass
class EvaluationResult:
    """Single evaluation result."""

    llm_name: str
    strategy: str
    example_id: int
    puzzle: list[int]
    expected: list[int]
    predicted: list[int] | None
    exact_match: bool
    partial_accuracy: float
    error: str | None = None


def create_llm_client(llm_name: str):
    """
    Create a LangChain chat model for the specified LLM.

    Args:
        llm_name: Name of the LLM from LLMApiConfig

    Returns:
        LangChain chat model with structured output configured
    """
    config = LLMApiConfig()
    llm_config = getattr(config, llm_name)

    # Claude Opus uses native Anthropic API
    if llm_name == "claude_opus_4_5":
        llm = ChatAnthropic(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            temperature=1,
            max_tokens=20000,
            # thinking={"type": "enabled", "budget_tokens": 1024},
            effort="high",
        )
    # Ollama models (Qwen via Ollama)
    elif llm_config.get("provider") == "ollama":
        llm = ChatOllama(
            model=llm_config["model"],
            temperature=1,
            max_tokens=20000,
        )
    else:
        # All others use OpenAI-compatible API (Gemini, OpenRouter)
        llm = ChatOpenAI(
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            temperature=1,
            max_tokens=20000,
        )

    return llm.with_structured_output(SudokuGrid)


def load_test_examples() -> list[tuple[list[int], list[int], int]]:
    """
    Load fixed test examples from HuggingFace dataset.

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


def build_messages(puzzle: list[int], strategy: Literal["zero_shot", "few_shot"]) -> list:
    """
    Build LangChain messages for the given puzzle and strategy.

    Args:
        puzzle: Flattened 81-element sudoku puzzle
        strategy: Either "zero_shot" or "few_shot"

    Returns:
        List of LangChain messages
    """
    prompts = SudokuPrompts()

    if strategy == "zero_shot":
        system_content = prompts.task_description
    else:  # few_shot
        system_content = prompts.few_shot_prompt

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=prompts.user_message_template.format(SUDOKU_GRID=str(puzzle))),
    ]


# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 60  # Wait 60s between retries for rate limiting


def evaluate_single(llm_client, puzzle: list[int], solution: list[int], example_id: int, llm_name: str, strategy: str) -> EvaluationResult:
    """
    Evaluate a single puzzle with the given LLM and strategy.

    Args:
        llm_client: LangChain structured LLM client
        puzzle: Input puzzle (81 integers)
        solution: Expected solution (81 integers)
        example_id: ID of the test example
        llm_name: Name of the LLM
        strategy: Prompt strategy used

    Returns:
        EvaluationResult with metrics
    """
    import time

    messages = build_messages(puzzle, strategy)

    if llm_name == "claude_opus_4_5":
        MAX_RETRIES = 5
    else:
        MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES):
        try:
            result = llm_client.invoke(messages)
            predicted = result.grid

            exact_match = calculate_exact_accuracy(predicted, solution)
            partial_acc = calculate_partial_accuracy(predicted, solution)

            return EvaluationResult(
                llm_name=llm_name,
                strategy=strategy,
                example_id=example_id,
                puzzle=puzzle,
                expected=solution,
                predicted=predicted,
                exact_match=exact_match,
                partial_accuracy=partial_acc,
            )
        except Exception as e:
            error_str = str(e)

            if attempt < MAX_RETRIES - 1:
                print(f"Error: {error_str}")
                print(f" [RETRY {attempt + 1}/{MAX_RETRIES}] Rate limited, waiting {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            # Return error result if no more retries or not a rate limit error
            return EvaluationResult(
                llm_name=llm_name,
                strategy=strategy,
                example_id=example_id,
                puzzle=puzzle,
                expected=solution,
                predicted=None,
                exact_match=False,
                partial_accuracy=0.0,
                error=error_str,
            )

    # Should never reach here, but just in case
    return EvaluationResult(
        llm_name=llm_name,
        strategy=strategy,
        example_id=example_id,
        puzzle=puzzle,
        expected=solution,
        predicted=None,
        exact_match=False,
        partial_accuracy=0.0,
        error="Max retries exceeded",
    )


def save_intermediate_results(results: list[EvaluationResult], llm_name: str, output_dir: Path) -> Path:
    """
    Save intermediate results for a single LLM to disk.

    Args:
        results: List of evaluation results for one LLM
        llm_name: Name of the LLM
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"intermediate_{llm_name}.json"

    data = {
        "llm_name": llm_name,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def load_intermediate_results(llm_name: str, output_dir: Path) -> list[dict] | None:
    """
    Load intermediate results for a single LLM if they exist.

    Args:
        llm_name: Name of the LLM
        output_dir: Directory to look for results

    Returns:
        List of result dicts or None if not found
    """
    filepath = output_dir / f"intermediate_{llm_name}.json"

    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        return data["results"]

    return None


def evaluate_llm(llm_name: str, examples: list, output_dir: Path, force: bool = False) -> list[EvaluationResult]:
    """
    Evaluate a single LLM on all examples with both strategies.

    Args:
        llm_name: Name of the LLM to evaluate
        examples: List of (puzzle, solution, example_id) tuples
        output_dir: Directory to save intermediate results
        force: If True, re-run even if intermediate results exist

    Returns:
        List of EvaluationResults for this LLM
    """
    # Check for existing results
    if not force:
        existing = load_intermediate_results(llm_name, output_dir)
        if existing:
            print(f"  [SKIP] Found existing results for {llm_name}, loading from cache")
            return [
                EvaluationResult(
                    llm_name=r["llm_name"],
                    strategy=r["strategy"],
                    example_id=r["example_id"],
                    puzzle=r["puzzle"],
                    expected=r["expected"],
                    predicted=r["predicted"],
                    exact_match=r["exact_match"],
                    partial_accuracy=r["partial_accuracy"],
                    error=r.get("error"),
                )
                for r in existing
            ]

    # Create LLM client
    try:
        llm_client = create_llm_client(llm_name)
    except Exception as e:
        print(f"  [ERROR] Failed to create client for {llm_name}: {e}")
        return []

    results = []

    for strategy in STRATEGIES:
        print(f"  Strategy: {strategy}")
        for puzzle, solution, example_id in examples:
            print(f"    Example {example_id}...", end=" ")
            result = evaluate_single(llm_client, puzzle, solution, example_id, llm_name, strategy)

            if result.error:
                print(f"ERROR: {result.error}...")
            else:
                print(f"Results: {result}")

            results.append(result)

    # Save intermediate results
    save_intermediate_results(results, llm_name, output_dir)
    print(f"  [SAVED] Intermediate results for {llm_name}")

    return results


def merge_all_results(output_dir: Path) -> list[dict]:
    """
    Merge all intermediate results into a single list.

    Args:
        output_dir: Directory containing intermediate results

    Returns:
        List of all result dicts
    """
    all_results = []

    for llm_name in LLM_NAMES:
        results = load_intermediate_results(llm_name, output_dir)
        if results:
            all_results.extend(results)

    return all_results


def run_full_evaluation(output_dir: Path | None = None, force: bool = False) -> dict:
    """
    Run full evaluation across all LLMs and strategies.

    Args:
        output_dir: Directory to save results (default: runs/llm_evaluation_<timestamp>)
        force: If True, re-run all evaluations even if cached

    Returns:
        Dict with all results and aggregated metrics
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"llm_evaluation_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load test examples
    print("\nLoading test examples...")
    examples = load_test_examples()
    print(f"Loaded {len(examples)} examples with IDs: {[e[2] for e in examples]}")

    # Evaluate each LLM
    all_results = []
    for llm_name in LLM_NAMES:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {llm_name}")
        print("=" * 50)

        results = evaluate_llm(llm_name, examples, output_dir, force=force)
        all_results.extend(results)

    # Merge and aggregate
    print("\n" + "=" * 50)
    print("Aggregating results...")
    print("=" * 50)

    from reasoning_lab.llm.metrics import aggregate_metrics

    result_dicts = [asdict(r) for r in all_results]
    aggregated = aggregate_metrics(result_dicts)

    # Save final results
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "test_sample_ids": TEST_SAMPLE_IDS,
        "llm_names": LLM_NAMES,
        "strategies": STRATEGIES,
        "results": result_dicts,
        "aggregated": {f"{k[0]}_{k[1]}": v for k, v in aggregated.items()},
    }

    final_path = output_dir / "final_results.json"
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nFinal results saved to: {final_path}")

    return final_output


def print_summary_table(aggregated: dict) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'LLM':<25} {'Strategy':<12} {'Exact Acc':<12} {'Partial Acc':<12}")
    print("-" * 70)

    for key, metrics in sorted(aggregated.items()):
        if isinstance(key, tuple):
            llm_name, strategy = key
        else:
            # Handle string keys from JSON
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ["shot"]:
                # Handle few_shot, zero_shot
                llm_name = key.rsplit("_", 2)[0]
                strategy = "_".join(key.rsplit("_", 2)[1:])
            else:
                llm_name, strategy = key.rsplit("_", 1)

        print(f"{llm_name:<25} {strategy:<12} {metrics['exact_accuracy']:<12.3f} {metrics['partial_accuracy']:<12.3f}")

    print("=" * 70)


if __name__ == "__main__":
    results = run_full_evaluation()
    print_summary_table(results["aggregated"])
