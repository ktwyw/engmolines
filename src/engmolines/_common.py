"""Shared result type."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    """Result of a method-of-lines integration: times, states (one row per time), and statistics."""

    t: np.ndarray
    y: np.ndarray
    steps: int = 0
    rejected: int = 0
    evaluations: int = 0
    newton_iterations: int = 0
    jacobian_evaluations: int = 0
    message: str = "ok"
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"Solution(t in [{self.t[0]:g}, {self.t[-1]:g}], {len(self.t)} outputs, state size {self.y.shape[1]}, "
            f"steps={self.steps}, rejected={self.rejected}, evaluations={self.evaluations}, "
            f"newton={self.newton_iterations}, jacobians={self.jacobian_evaluations}, {self.message!r})"
        )
