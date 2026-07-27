"""Execution-backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from swarm_seek.space import Space
from swarm_seek.types import FloatArray


@runtime_checkable
class BackendProtocol(Protocol):
    """Array kernels used by variant strategies."""

    name: str

    def generate_candidates(
        self,
        population: FloatArray,
        partner_indices: NDArray[np.integer],
        phi: FloatArray,
        *,
        space: Space,
    ) -> FloatArray:
        """Produce repaired neighbor candidates.

        Implements the shared arithmetic
        ``v = x + phi * (x - x_partner)`` followed by ``space.repair``.
        """
        ...
