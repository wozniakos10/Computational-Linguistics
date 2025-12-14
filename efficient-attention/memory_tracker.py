from typing import Dict

import torch


class MemoryTracker:
    """Simple memory tracker for training steps."""

    def __init__(self, device: torch.device, track_every_n_steps: int = 10):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.track_every_n_steps = track_every_n_steps
        self.current_step = 0
        self.should_track = False
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.step_max_memory = 0
        self.max_forward_memory = 0
        self.max_backward_memory = 0

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_step(self):
        """Mark start of training step."""
        self.should_track = self.current_step % self.track_every_n_steps == 0
        self.current_step += 1

        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            self.step_max_memory = torch.cuda.max_memory_allocated(self.device)

    def start_forward(self):
        """Mark start of forward pass."""
        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

    def mark_forward(self):
        """Record memory after forward pass."""
        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            self.max_forward_memory = torch.cuda.max_memory_allocated(self.device)

    def start_backward(self):
        """Mark start of backward pass."""
        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

    def mark_backward(self):
        """Record memory after backward pass."""
        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            self.max_backward_memory = torch.cuda.max_memory_allocated(self.device)

    def mark_entire_step(self):
        """Mark end of training step and compute memory usage."""
        if self.is_cuda and self.should_track:
            torch.cuda.synchronize()
            step_end_memory = torch.cuda.max_memory_allocated(self.device)
            self.step_max_memory = max(step_end_memory, self.step_max_memory, self.max_forward_memory, self.max_backward_memory)
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_stats(self) -> Dict[str, float] | None:
        """Get current stats in GB. Returns None if this step wasn't tracked."""
        if not self.should_track:
            return None

        return {
            "step_max_memory_gb": self.step_max_memory / (1024**3),
            "max_forward_memory_gb": self.max_forward_memory / (1024**3),
            "max_backward_memory_gb": self.max_backward_memory / (1024**3),
        }
