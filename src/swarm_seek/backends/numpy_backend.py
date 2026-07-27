"""NumPy execution backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swarm_seek.backends._validate import validate_generate_inputs
from swarm_seek.space import Space
from swarm_seek.types import FloatArray


class NumpyBackend:
    """Vectorized NumPy implementation of shared ABC array primitives.

    Candidate generation follows the neighbor search arithmetic used by
    Artificial Bee Colony (Karaboga & Basturk, 2007):

    ``v = x + phi * (x - x_k)``,

    then projects with ``space.repair``. Dimensions with ``phi == 0`` are
    left unchanged so variants can mutate a single coordinate.
    """

    name = "numpy"

    def generate_candidates(
        self,
        population: FloatArray,
        partner_indices: NDArray[np.integer],
        phi: FloatArray,
        *,
        space: Space,
    ) -> FloatArray:
        """Return repaired candidates without mutating ``population``."""
        pop, partners, weights = validate_generate_inputs(
            population, partner_indices, phi
        )
        partner_rows = pop[partners]
        candidates = pop + weights * (pop - partner_rows)
        return space.repair(candidates)
