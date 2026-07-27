"""Numba execution backend (optional ``swarm-seek[numba]`` extra)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swarm_seek.backends._validate import validate_generate_inputs
from swarm_seek.space import Space
from swarm_seek.types import FloatArray

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - exercised via missing-extra test
    _NUMBA_IMPORT_ERROR = exc
    njit = None  # type: ignore[assignment]
else:
    _NUMBA_IMPORT_ERROR = None


def _require_numba() -> None:
    if _NUMBA_IMPORT_ERROR is not None:
        msg = (
            "The Numba backend requires the optional dependency 'numba'. "
            "Install with: pip install 'swarm-seek[numba]'."
        )
        raise ImportError(msg) from _NUMBA_IMPORT_ERROR


if njit is not None:

    @njit(cache=True)
    def _arith(
        pop: np.ndarray,
        partners: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        n_pop, n_dim = pop.shape
        out = np.empty((n_pop, n_dim), dtype=np.float64)
        for i in range(n_pop):
            k = partners[i]
            for j in range(n_dim):
                out[i, j] = pop[i, j] + weights[i, j] * (pop[i, j] - pop[k, j])
        return out

else:  # pragma: no cover

    def _arith(
        pop: np.ndarray,
        partners: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        raise ImportError("numba is not installed")


class NumbaBackend:
    """Numba-JIT continuous neighbor arithmetic with ``space.repair``.

    Implements the same shared primitive as :class:`NumpyBackend`:

    ``v = x + phi * (x - x_partner)``,

    then ``space.repair``. Requires ``pip install 'swarm-seek[numba]'``.
    """

    name = "numba"

    def __init__(self) -> None:
        _require_numba()

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
        candidates = _arith(pop, partners, weights)
        return space.repair(candidates)
