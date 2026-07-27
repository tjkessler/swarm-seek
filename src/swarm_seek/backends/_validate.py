"""Shared input validation for continuous backends."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import BoundsError
from swarm_seek.types import FloatArray


def validate_generate_inputs(
    population: FloatArray,
    partner_indices: NDArray[np.integer],
    phi: FloatArray,
) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Validate and normalize arrays for ``generate_candidates``."""
    pop = np.asarray(population, dtype=np.float64)
    partners = np.asarray(partner_indices, dtype=np.int64)
    weights = np.asarray(phi, dtype=np.float64)

    if pop.ndim != 2:
        msg = f"population must be 2-D (n, d); got shape {pop.shape}."
        raise BoundsError(msg)
    n_pop, _n_dim = pop.shape
    if partners.shape != (n_pop,):
        msg = f"partner_indices must have shape ({n_pop},); got {partners.shape}."
        raise BoundsError(msg)
    if weights.shape != pop.shape:
        msg = f"phi must have shape {pop.shape}; got {weights.shape}."
        raise BoundsError(msg)
    if n_pop == 0:
        msg = "population must contain at least one row."
        raise BoundsError(msg)
    return pop, partners, weights
