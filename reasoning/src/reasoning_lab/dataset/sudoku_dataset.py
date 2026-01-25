from typing import Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from reasoning_lab.trm import common


class SudokuDataset(Dataset):
    def __init__(self, split: str = "train", start_idx: int = 0, end_idx: Optional[int] = None, augment: bool = False):
        """
        Args:
            split: "train" or "test" (mapped to dataset splits)
            start_idx: Start index for selecting subset (inclusive)
            end_idx: End index for selecting subset (exclusive), None for all
            augment: Whether to apply augmentations (digit permutation + dihedral)
        """
        self.augment_data = augment

        # Load dataset from HuggingFace
        # sapientinc/sudoku-extreme has 'train', 'test' splits
        ds = load_dataset("sapientinc/sudoku-extreme", split=split)

        # Select subset
        total_len = len(ds)
        start = start_idx
        end = end_idx if end_idx is not None else total_len
        # Ensure bounds
        end = min(end, total_len)

        if start < end:
            ds = ds.select(range(start, end))
        else:
            # Handle empty case or error? For now assume valid range
            ds = ds.select([])

        self.data = []
        for item in ds:
            # Parse 'question' and 'answer' strings into numpy arrays
            # Replace '.' with '0' for blanks
            q_str = item["question"].replace(".", "0")
            a_str = item["answer"].replace(".", "0")

            puzzle = np.array([int(c) for c in q_str], dtype=np.int64).reshape(9, 9)
            solution = np.array([int(c) for c in a_str], dtype=np.int64).reshape(9, 9)
            self.data.append((puzzle, solution))

        self.metadata = common.PuzzleDatasetMetadata(
            pad_id=0,  # Not strictly used as we have fixed length
            ignore_label_id=-100,  # Standard PyTorch ignore index
            blank_identifier_id=0,
            vocab_size=10,  # digits 0-9
            seq_len=81,
            num_puzzle_identifiers=0,  # Not used
            total_groups=1,
            mean_puzzle_examples=float(len(self.data)),
            total_puzzles=len(self.data),
            sets=[split],
        )

    def __len__(self):
        # We produce new augmentations on the fly, so length is effectively infinite/arbitrary
        # or we could define a fixed epoch size.
        # For simplicity in this task, let's say 1 epoch = base_examples * 10
        if self.augment_data:
            return len(self.data) * 10
        return len(self.data)

    def _augment(self, puzzle: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Random Digit Permutation (1-9)
        # Create a random mapping for 1-9, keep 0 map to 0
        digit_map = np.arange(10)
        perm = np.random.permutation(np.arange(1, 10))
        digit_map[1:] = perm

        puzzle = digit_map[puzzle]
        solution = digit_map[solution]

        # 2. Dihedral Transformation using common.py
        # There are 8 symmetries. Pick one randomly.
        tid = np.random.randint(0, 8)
        puzzle = common.dihedral_transform(puzzle, tid)
        solution = common.dihedral_transform(solution, tid)

        return puzzle, solution

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map linear index to base example index
        base_idx = idx % len(self.data)
        puzzle, solution = self.data[base_idx]

        # Copy to avoid modifying original
        puzzle = puzzle.copy()
        solution = solution.copy()

        if self.augment_data:
            puzzle, solution = self._augment(puzzle, solution)

        # Flatten and convert to tensor
        puzzle_flat = torch.from_numpy(puzzle.flatten()).long()
        solution_flat = torch.from_numpy(solution.flatten()).long()

        return puzzle_flat, solution_flat


if __name__ == "__main__":
    dataset = SudokuDataset(split="train", augment=False)
    print(dataset.metadata)
    for idx, input_data in enumerate(dataset):
        puzzle, solution = input_data
        print(f"Example {idx}:")
        print("Input:")
        print(puzzle.numpy())
        print("Output:")
        print(solution.numpy())
        print("\n\n")
        if idx >= 10:
            break
