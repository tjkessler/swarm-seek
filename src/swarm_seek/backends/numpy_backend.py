"""NumPy execution backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import BoundsError
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
        pop = np.asarray(population, dtype=np.float64)
        partners = np.asarray(partner_indices)
        weights = np.asarray(phi, dtype=np.float64)

        if pop.ndim != 2:
            msg = f"population must be 2-D (n, d); got shape {pop.shape}."
            raise BoundsError(msg)
        n_pop, n_dim = pop.shape
        if partners.shape != (n_pop,):
            msg = f"partner_indices must have shape ({n_pop},); got {partners.shape}."
            raise BoundsError(msg)
        if weights.shape != pop.shape:
            msg = f"phi must have shape {pop.shape}; got {weights.shape}."
            raise BoundsError(msg)
        if n_pop == 0:
            msg = "population must contain at least one row."
            raise BoundsError(msg)

        partner_rows = pop[partners]
        candidates = pop + weights * (pop - partner_rows)
        return space.repair(candidates)
