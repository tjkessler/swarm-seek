"""Candidate solution type."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm_seek.errors import SolutionValidationError
from swarm_seek.types._aliases import FloatArray


@dataclass(frozen=True, slots=True)
class Solution:
    """A single candidate in the search space.

    Parameters
    ----------
    x
        Decision-variable vector with shape ``(n_dim,)``.
    fitness
        Objective value associated with ``x``.
    """

    x: FloatArray
    fitness: float

    def __post_init__(self) -> None:
        array = np.asarray(self.x, dtype=np.float64)
        if array.ndim != 1:
            msg = f"Solution.x must be 1-D; got shape {array.shape}."
            raise SolutionValidationError(msg)
        if array.size == 0:
            msg = "Solution.x must be non-empty."
            raise SolutionValidationError(msg)
        if not np.all(np.isfinite(array)):
            msg = "Solution.x must contain only finite values."
            raise SolutionValidationError(msg)
        try:
            fitness = float(self.fitness)
        except (TypeError, ValueError) as exc:
            msg = "Solution.fitness must be a real number."
            raise SolutionValidationError(msg) from exc
        if not np.isfinite(fitness):
            msg = "Solution.fitness must be finite."
            raise SolutionValidationError(msg)
        object.__setattr__(self, "x", array)
        object.__setattr__(self, "fitness", fitness)
