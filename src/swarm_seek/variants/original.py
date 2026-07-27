"""Original Artificial Bee Colony (ABC) update rules.

Implements the employed, onlooker, and scout operators of the Artificial Bee
Colony algorithm introduced by Karaboga (2005) and elaborated by Karaboga &
Basturk (2007). Neighbor generation uses the shared backend primitive

``v_{ij} = x_{ij} + \\phi_{ij} (x_{ij} - x_{kj})``

with a single randomly chosen dimension ``j`` and partner ``k \\neq i``, and
``\\phi_{ij} \\in [-1, 1]`` (Karaboga & Basturk, 2007, employed/onlooker
neighbor search). Bound handling is delegated to ``space.repair`` (clip) via
the backend.

Parameter mapping
-----------------
limit
    Abandonment limit (``limit`` in Karaboga & Basturk, 2007).
sense
    ``\"minimize\"`` (default) or ``\"maximize\"`` — controls nectar / selection
    fitness used by onlookers.

Notes
-----
Greedy selection and trial-counter increments after employed/onlooker
proposals are owned by the colony engine, not this strategy. ``scout_step``
resets trials for abandoned sources.

References
----------
Karaboga, D. (2005). *An Idea Based on Honey Bee Swarm for Numerical
Optimization.* Technical Report TR06, Erciyes University.

Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for
numerical function optimization: artificial bee colony (ABC) algorithm.
*Journal of Global Optimization*, 39(3), 459–471.
https://doi.org/10.1007/s10898-007-9149-x
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import SwarmSeekError
from swarm_seek.types import FloatArray
from swarm_seek.variants._helpers import (
    abandon_exhausted,
    partner_indices,
    require_population,
    require_sense,
    selection_probabilities,
    sparse_phi,
)
from swarm_seek.variants.registry import register_variant

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space

# Re-export helpers under historical private names for existing tests.
_partner_indices = partner_indices
_sparse_phi = sparse_phi


def _nectar(fitness: FloatArray, *, sense: str) -> FloatArray:
    from swarm_seek.variants._helpers import nectar

    return nectar(fitness, sense=sense)


def _neighbor_candidates(
    population: FloatArray,
    rng: np.random.Generator,
    *,
    backend: BackendProtocol,
    space: Space,
) -> FloatArray:
    pop = require_population(population, label="Original ABC")
    n_pop, n_dim = pop.shape
    partners = _partner_indices(n_pop, rng)
    phi = _sparse_phi(n_pop, n_dim, rng)
    return backend.generate_candidates(pop, partners, phi, space=space)


class OriginalABC:
    """Original ABC employed / onlooker / scout strategy."""

    name = "original"
    citation = (
        "Karaboga (2005) TR06; Karaboga & Basturk (2007) "
        "J. Glob. Optim. 39:459–471, doi:10.1007/s10898-007-9149-x"
    )

    def employed_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> FloatArray:
        """Propose one neighbor candidate per food source (employed bees)."""
        del fitness, params  # unused in Original employed phase
        return _neighbor_candidates(population, rng, backend=backend, space=space)

    def onlooker_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> FloatArray:
        """Propose onlooker neighbors via fitness-proportional source selection.

        Returns a ``(n_pop, d)`` array aligned to food-source indices. Each of
        ``n_pop`` onlookers roulette-selects a source and overwrites that row
        with a neighbor proposal. Sources never selected remain copies of the
        current population (no evaluation needed).
        """
        pop = require_population(population, label="Original ABC")
        n_pop, n_dim = pop.shape
        fit = np.asarray(fitness, dtype=np.float64)
        if fit.shape != (n_pop,):
            msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
            raise SwarmSeekError(msg)

        sense = require_sense(params)
        probs = selection_probabilities(fit, sense=sense)
        selected = rng.choice(n_pop, size=n_pop, replace=True, p=probs)

        candidates = pop.copy()
        for source_idx in selected:
            idx = int(source_idx)
            partner = int(rng.integers(0, n_pop - 1))
            if partner >= idx:
                partner += 1
            dim = int(rng.integers(0, n_dim))
            phi = np.zeros((n_pop, n_dim), dtype=np.float64)
            phi[idx, dim] = float(rng.uniform(-1.0, 1.0))
            partners = np.zeros(n_pop, dtype=np.int64)
            partners[idx] = partner
            batch = backend.generate_candidates(pop, partners, phi, space=space)
            candidates[idx] = batch[idx]
        return candidates

    def scout_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        trials: NDArray[np.integer],
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> tuple[FloatArray, NDArray[np.integer]]:
        """Reinitialize food sources whose trial counter reached ``limit``."""
        del fitness, backend
        return abandon_exhausted(population, trials, rng, space=space, params=params)


register_variant("original", OriginalABC, replace=True)
