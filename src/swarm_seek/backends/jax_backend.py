"""JAX execution backend (optional ``swarm-seek[jax]`` extra)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from swarm_seek.backends._validate import validate_generate_inputs
from swarm_seek.space import Space
from swarm_seek.types import FloatArray

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised via missing-extra test
    _JAX_IMPORT_ERROR = exc
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
else:
    _JAX_IMPORT_ERROR = None


def _require_jax() -> None:
    if _JAX_IMPORT_ERROR is not None:
        msg = (
            "The JAX backend requires the optional dependency 'jax'. "
            "Install with: pip install 'swarm-seek[jax]'."
        )
        raise ImportError(msg) from _JAX_IMPORT_ERROR


class JaxBackend:
    """JAX continuous neighbor arithmetic with host ``float64`` boundary.

    Implements the same shared primitive as :class:`NumpyBackend`:

    ``v = x + phi * (x - x_partner)``,

    then ``space.repair`` on host NumPy arrays. Requires
    ``pip install 'swarm-seek[jax]'``. Never selected by ``backend="auto"``.
    """

    name = "jax"

    def __init__(self) -> None:
        _require_jax()
        # Prefer float64 at the colony boundary (design Q9).
        jax.config.update("jax_enable_x64", True)

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
        pop_j = jnp.asarray(pop)
        partners_j = jnp.asarray(partners)
        weights_j = jnp.asarray(weights)
        partner_rows = pop_j[partners_j]
        candidates_j = pop_j + weights_j * (pop_j - partner_rows)
        candidates = np.asarray(candidates_j, dtype=np.float64)
        return space.repair(candidates)
