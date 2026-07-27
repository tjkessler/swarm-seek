"""Search-space protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from swarm_seek.types import FloatArray


@runtime_checkable
class Space(Protocol):
    """Contract for candidate representation and bound handling."""

    @property
    def n_dim(self) -> int:
        """Number of decision variables."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Draw ``n`` candidates using ``rng``."""
        ...

    def repair(self, x: FloatArray) -> FloatArray:
        """Project candidates into the feasible region."""
        ...

    def validate(self, x: FloatArray) -> None:
        """Raise if candidates are not feasible / correctly shaped."""
        ...
