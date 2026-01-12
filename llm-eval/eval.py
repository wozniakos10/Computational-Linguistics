import json
import time

from ollama import chat

from logger import get_configured_logger
from models import AvailableModels
from tasks import (
    CodeGeneration,
    CommonSenseReasoning,
    CreativeWriting,
    EthicalReasoning,
    FactualKnowledge,
    InstructionsFollowing,
    LanguageUnderstanding,
    LogicalReasoning,
    MathematicalProblemSolving,
    ReadingComprehension,
    Task,
)

logger = get_configured_logger(__name__, log_file="logs/evaluation.log")


def evaluate_model_on_task(model_name: AvailableModels, task: Task) -> dict[str, str]:
    """Function to evaluate a given model on a specified task using zero-shot, few-shot, and chain-of-thought prompting."""
    response_zero_shot = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": task.task_description,
            },
        ],
    )

    response_few_shot = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": task.few_shot_prompt,
            },
        ],
    )

    if model_name != "qwen3:14b":
        response_cot = chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": task.cot_prompt,
                },
            ],
        )
    else:
        response_cot = None

    return {
        "zero_shot": response_zero_shot.message.content,
        "few_shot": response_few_shot.message.content,
        "cot": response_cot.message.content if response_cot else None,
    }


def run_evaluation_pipeline():
    """Function to run the evaluation pipeline for all models and tasks."""
    models_to_evaluate: list[AvailableModels] = ["qwen2.5:1.5b", "qwen3:14b"]
    tasks_to_evaluate: list[Task] = [
        LogicalReasoning(),
        InstructionsFollowing(),
        CreativeWriting(),
        CodeGeneration(),
        ReadingComprehension(),
        CommonSenseReasoning(),
        LanguageUnderstanding(),
        FactualKnowledge(),
        MathematicalProblemSolving(),
        EthicalReasoning(),
    ]

    results = {}
    for task in tasks_to_evaluate:
        results[task.task_name] = {}
        results[task.task_name]["task_description"] = task.task_description

        for model in models_to_evaluate:
            start = time.time()
            logger.info(
                f"Starting evaluation for task '{task.task_name}' with model '{model}'."
            )
            evaluation_result = evaluate_model_on_task(model, task)
            results[task.task_name][model] = evaluation_result
            stop = time.time()
            logger.info(
                f"Completed evaluation for task '{task.task_name}' with model '{model}' in {stop - start:.2f} seconds."
            )
        task.save_results(results[task.task_name])

    return results


if __name__ == "__main__":
    evaluation_results = run_evaluation_pipeline()
    with open("logs/final_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
