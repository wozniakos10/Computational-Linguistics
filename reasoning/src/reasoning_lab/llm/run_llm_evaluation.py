#!/usr/bin/env python3
"""
CLI runner for LLM sudoku evaluation.

Usage:
    uv run python -m reasoning_lab.llm.run_llm_evaluation [--force] [--output-dir DIR]
"""

import argparse
from pathlib import Path

from reasoning_lab.llm.llm_evaluator import (
    LLM_NAMES,
    STRATEGIES,
    TEST_SAMPLE_IDS,
    print_summary_table,
    run_full_evaluation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on sudoku solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration:
  LLMs to evaluate: {", ".join(LLM_NAMES)}
  Strategies: {", ".join(STRATEGIES)}
  Test sample IDs: {TEST_SAMPLE_IDS}

Examples:
  # Run full evaluation (uses cached results if available)
  uv run python -m reasoning_lab.llm.run_llm_evaluation

  # Force re-run all evaluations
  uv run python -m reasoning_lab.llm.run_llm_evaluation --force

  # Specify output directory
  uv run python -m reasoning_lab.llm.run_llm_evaluation --output-dir runs/my_evaluation
""",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run evaluations even if cached results exist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: runs/llm_evaluation_<timestamp>)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LLM Sudoku Evaluation")
    print("=" * 70)
    print(f"LLMs: {', '.join(LLM_NAMES)}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Test samples: {TEST_SAMPLE_IDS}")
    print(f"Force re-run: {args.force}")
    print("=" * 70)

    try:
        results = run_full_evaluation(output_dir=args.output_dir, force=args.force)
        print_summary_table(results["aggregated"])
        print("\nEvaluation complete!")
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted. Intermediate results have been saved.")
        print("Run again without --force to resume from cached results.")
    except Exception as e:
        print(f"\n\nEvaluation failed: {e}")
        print("Check intermediate results in the output directory.")
        raise


if __name__ == "__main__":
    main()
