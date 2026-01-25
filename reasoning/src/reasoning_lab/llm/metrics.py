"""
Metrics calculation for LLM sudoku evaluation.
"""


def calculate_exact_accuracy(predicted: list[int], expected: list[int]) -> bool:
    """
    Returns True if sudoku is completely solved correctly.
    All 81 cells must match exactly.
    """
    if predicted is None or len(predicted) != 81:
        return False
    return predicted == expected


def calculate_partial_accuracy(predicted: list[int], expected: list[int]) -> float:
    """
    Returns fraction of correct cells (out of 81).
    E.g., if 7 cells are correct, returns 7/81 ≈ 0.086.
    """
    if predicted is None or len(predicted) != 81:
        return 0.0

    correct_count = sum(1 for p, e in zip(predicted, expected) if p == e)
    return correct_count / 81


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Aggregate metrics by LLM and strategy.

    Args:
        results: List of evaluation result dicts with keys:
            - llm_name, strategy, exact_match, partial_accuracy

    Returns:
        Dict with structure:
        {
            (llm_name, strategy): {
                "exact_accuracy": float,  # fraction of exact matches
                "partial_accuracy": float,  # average partial accuracy
                "count": int
            }
        }
    """
    from collections import defaultdict

    aggregated = defaultdict(lambda: {"exact_matches": 0, "partial_sum": 0.0, "count": 0})

    for result in results:
        key = (result["llm_name"], result["strategy"])
        aggregated[key]["exact_matches"] += int(result["exact_match"])
        aggregated[key]["partial_sum"] += result["partial_accuracy"]
        aggregated[key]["count"] += 1

    # Calculate averages
    output = {}
    for key, data in aggregated.items():
        count = data["count"]
        output[key] = {
            "exact_accuracy": data["exact_matches"] / count if count > 0 else 0.0,
            "partial_accuracy": data["partial_sum"] / count if count > 0 else 0.0,
            "count": count,
        }

    return output
