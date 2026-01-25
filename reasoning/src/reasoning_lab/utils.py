"""
Shared utility functions for TRM Sudoku training and inference.
"""

import numpy as np
import torch
from einops import unpack
from tiny_recursive_model import TinyRecursiveModel


def range_from_one(n):
    """Range starting from 1 instead of 0."""
    return range(1, n + 1)


def is_empty(t):
    """Check if tensor has no elements."""
    return t.numel() == 0


def format_sudoku_grid(flat_grid):
    """Formats a flattened 81-element tensor/array into a 9x9 grid string."""
    grid = np.array(flat_grid).reshape(9, 9)
    s = ""
    for r in range(9):
        if r > 0 and r % 3 == 0:
            s += "------+-------+------\n"
        row = ""
        for c in range(9):
            if c > 0 and c % 3 == 0:
                row += "| "
            row += str(grid[r, c]) + " "
        s += row + "\n"
    return s


# --- Monkeypatch for TinyRecursiveModel.predict to fix GPU device mismatch ---
# Actualy same logic from tiny-resoning-model repated, but fix gpu device bug
@torch.no_grad()
def _fixed_predict(self, seq, halt_prob_thres=0.5, max_deep_refinement_steps=12):
    batch = seq.shape[0]

    inputs, packed_shape = self.embed_inputs_with_registers(seq)

    # initial outputs and latents
    outputs, latents = self.get_initial()

    # active batch indices, the step it exited at, and the final output predictions
    active_batch_indices = torch.arange(batch, device=self.device, dtype=torch.long)

    preds = []
    exited_step_indices = []
    exited_batch_indices = []

    for step in range_from_one(max_deep_refinement_steps):
        is_last = step == max_deep_refinement_steps

        outputs, latents = self.deep_refinement(inputs, outputs, latents)

        halt_prob = self.to_halt_pred(outputs).sigmoid()

        should_halt = (halt_prob >= halt_prob_thres) | is_last

        if not should_halt.any():
            continue

        # maybe remove registers
        registers, outputs_for_pred = unpack(outputs, packed_shape, "b * d")

        # append to exited predictions
        pred = self.to_pred(outputs_for_pred[should_halt])
        preds.append(pred)

        # append the step at which early halted
        exited_step_indices.extend([step] * should_halt.sum().item())

        # append indices for sorting back
        exited_batch_indices.append(active_batch_indices[should_halt])

        if is_last:
            continue

        # ready for next round
        inputs = inputs[~should_halt]
        outputs = outputs[~should_halt]
        latents = latents[~should_halt]
        active_batch_indices = active_batch_indices[~should_halt]

        if is_empty(outputs):
            break

    preds = torch.cat(preds).argmax(dim=-1)
    # FIX: Ensure tensor is created on the correct device
    exited_step_indices = torch.tensor(exited_step_indices, device=self.device)

    exited_batch_indices = torch.cat(exited_batch_indices)
    sort_indices = exited_batch_indices.argsort(dim=-1)

    return preds[sort_indices], exited_step_indices[sort_indices]


def apply_trm_predict_fix():
    """Apply the fixed predict method to TinyRecursiveModel. Call once at module import."""
    TinyRecursiveModel.predict = _fixed_predict


# Apply monkeypatch on import
apply_trm_predict_fix()
