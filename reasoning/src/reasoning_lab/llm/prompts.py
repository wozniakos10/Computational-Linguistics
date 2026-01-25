import json
import os
from abc import ABC

from langchain_core.prompts import PromptTemplate


class Task(ABC):
    """Abstract base class for evaluation tasks."""

    def __init__(self, task_description: str, examples: list, task_name: str):
        self._task_description = task_description
        self.examples = examples
        self.task_name = task_name

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def cot_prompt(self) -> str:
        return f"{self.task_description}\nLet's think step by step before answering."

    @property
    def few_shot_prompt(self) -> str:
        examples_str = ""
        for idx, example in enumerate(self.examples, 1):
            examples_str += f"Example {idx}:\n\n{example}\n\n"
        return f"{self.task_description}\nHere are some examples:\n\n{examples_str}"

    def save_results(self, results: dict) -> None:
        """Saves the evaluation results to a specified file."""
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", f"{self.task_name}_results.json")
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4, encoding="utf-8")


class SudokuPrompts(Task):
    def __init__(self):
        task_description = """You are a highly intelligent AI specialized in solving Sudoku puzzles. Your task is to fill in the missing numbers in a 9x9 Sudoku grid according to the game's rules. 
The grid is represented as a flattened 9x9 list of 81 integers (0-9), row by row from top-left to bottom-right. '0' indicates an empty cell. Return only the completed grid in the same format, do not include any explainations or additional text."""
        task_name = "sudoku_solver"
        examples = [
            """Input: [5, 0, 0, 0, 2, 7, 0, 0, 9, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 3, 0, 0, 0, 9, 2, 0, 6, 0, 8, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 7, 0, 0, 2, 9, 0, 8, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 3, 6, 0, 0]\n
Output: [5, 8, 3, 4, 2, 7, 1, 6, 9, 9, 7, 4, 1, 3, 6, 5, 2, 8, 2, 1, 6, 8, 5, 9, 3, 7, 4, 7, 9, 2, 3, 6, 4, 8, 5, 1, 3, 5, 1, 2, 9, 8, 7, 4, 6, 6, 4, 8, 7, 1, 5, 2, 9, 3, 8, 6, 5, 9, 7, 1, 4, 3, 2, 1, 3, 7, 6, 4, 2, 9, 8, 5, 4, 2, 9, 5, 8, 3, 6, 1, 7]""",
            """Input: [0, 2, 0, 1, 0, 0, 9, 0, 4, 0, 0, 0, 0, 2, 0, 6, 0, 0, 0, 0, 3, 0, 9, 4, 0, 0, 5, 0, 0, 4, 9, 7, 0, 0, 5, 0, 0, 1, 0, 0, 0, 2, 4, 0, 7, 0, 0, 0, 0, 4, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 9, 0, 0, 1, 0, 7, 0, 6]\n
Output: [8, 2, 7, 1, 6, 5, 9, 3, 4, 4, 5, 9, 3, 2, 8, 6, 7, 1, 1, 6, 3, 7, 9, 4, 8, 2, 5, 3, 8, 4, 9, 7, 6, 1, 5, 2, 6, 1, 5, 8, 3, 2, 4, 9, 7, 9, 7, 2, 5, 4, 1, 3, 6, 8, 2, 3, 6, 4, 8, 7, 5, 1, 9, 7, 4, 1, 6, 5, 9, 2, 8, 3, 5, 9, 8, 2, 1, 3, 7, 4, 6]""",
            """Input: [0, 7, 0, 4, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 5, 0, 6, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 9, 0, 8, 5, 0, 0, 9, 0, 0, 6, 6, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 9, 0, 6, 0, 2, 0, 0]\n
Output: [8, 7, 3, 4, 2, 6, 9, 1, 5, 5, 9, 1, 3, 8, 7, 6, 4, 2, 4, 2, 6, 5, 9, 1, 7, 3, 8, 9, 5, 7, 6, 1, 4, 8, 2, 3, 2, 6, 4, 9, 3, 8, 1, 5, 7, 3, 1, 8, 7, 5, 2, 4, 6, 9, 1, 8, 5, 2, 4, 9, 3, 7, 6, 6, 4, 2, 8, 7, 3, 5, 9, 1, 7, 3, 9, 1, 6, 5, 2, 8, 4]""",
            """Input: [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 3, 7, 0, 1, 0, 0, 7, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 8, 0, 4, 0, 0, 0, 0, 7, 3, 0, 0, 1, 0, 9, 0, 0, 3, 0, 0, 5, 0, 7, 0, 2, 0, 0, 0, 1, 9, 6, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]\n
Output: [9, 6, 1, 5, 8, 4, 7, 3, 2, 8, 4, 5, 3, 7, 2, 1, 9, 6, 7, 2, 3, 6, 1, 9, 5, 4, 8, 2, 3, 7, 1, 5, 6, 9, 8, 4, 4, 5, 6, 8, 9, 7, 3, 2, 1, 1, 8, 9, 4, 2, 3, 6, 7, 5, 3, 7, 4, 2, 6, 5, 8, 1, 9, 6, 9, 8, 7, 4, 1, 2, 5, 3, 5, 1, 2, 9, 3, 8, 4, 6, 7]""",
            """Input: [0, 0, 5, 0, 8, 9, 0, 0, 0, 0, 6, 0, 0, 5, 7, 0, 8, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 1, 0, 0, 9, 8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 1, 0, 8, 7, 0, 0, 0, 0, 4, 0, 0, 6, 5, 0, 0, 0, 7, 0]\n
Output: [7, 3, 5, 1, 8, 9, 6, 4, 2, 4, 6, 2, 3, 5, 7, 1, 8, 9, 8, 9, 1, 6, 2, 4, 7, 5, 3, 3, 8, 4, 2, 7, 1, 9, 6, 5, 6, 5, 7, 4, 9, 3, 8, 2, 1, 2, 1, 9, 8, 6, 5, 4, 3, 7, 5, 7, 3, 9, 4, 8, 2, 1, 6, 1, 2, 8, 7, 3, 6, 5, 9, 4, 9, 4, 6, 5, 1, 2, 3, 7, 8]""",
        ]
        self.user_message_template = PromptTemplate(template="{SUDOKU_GRID}")
        super().__init__(task_description=task_description, examples=examples, task_name=task_name)


if __name__ == "__main__":
    sd = SudokuPrompts()
    print("Task Description:\n", sd.task_description)
    print(sd.few_shot_prompt)
    print(
        sd.user_message_template.format(
            SUDOKU_GRID="[0 7 0 4 0 0 0 1 0 5 0 0 0 0 0 0 4 2 0 0 0 0 0 1 7 0 0 0 5 0 6 0 0 0 2 0 0 0 4 0 0 8 0 0 0 3 0 0 0 5 0 0 0 9 0 8 5 0 0 9 0 0 6 6 0 0 8 0 0 0 0 0 0 3 9 0 6 0 2 0 0]"
        )
    )
