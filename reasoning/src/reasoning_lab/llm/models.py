from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from reasoning_lab.llm.config import cfg


def get_openai_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    return OpenAI(api_key=cfg.openai_api_key)


class SudokuGrid(BaseModel):
    grid: list[int] = Field(
        description="Flattened 9x9 Sudoku grid as 81 integers (0-9), row by row from top-left to bottom-right. 0 represents empty cells."
    )

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, v: list[int]) -> list[int]:
        if len(v) != 81:
            raise ValueError("Sudoku grid must contain exactly 81 integers.")
        for value in v:
            if not (0 <= value <= 9):
                raise ValueError("Sudoku grid values must be integers between 0 and 9.")
        return v


if __name__ == "__main__":
    sg = SudokuGrid(grid=[12] * 81)
