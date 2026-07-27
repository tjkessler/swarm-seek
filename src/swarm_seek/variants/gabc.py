"""Gbest-guided Artificial Bee Colony (GABC) update rules.

Implements the gbest-guided neighbor search of Zhu & Kwong (2010). Relative to
Original ABC, each mutated coordinate adds a pull toward the current global
best food source ``y``:

``v_{ij} = x_{ij} + \\phi_{ij} (x_{ij} - x_{kj}) + \\psi_{ij} (y_j - x_{ij})``

with ``\\phi_{ij} \\in [-1, 1]`` and ``\\psi_{ij} \\in [0, C]`` on a single
randomly chosen dimension ``j`` (Zhu & Kwong, 2010, GABC solution search
equation). Bound handling uses ``space.repair`` (clip) after the full update.

Parameter mapping
-----------------
limit
    Abandonment limit (shared with Original ABC).
sense
    ``\"minimize\"`` (default) or ``\"maximize\"`` — selects global best and
    onlooker nectar.
C
    Upper bound on ``\\psi`` (Zhu & Kwong’s ``C``); default ``1.5``.

Notes
-----
The shared NumPy backend primitive encodes only the Original ABC neighbor
term. GABC evaluates the full equation in-strategy, then repairs once, so the
gbest term is applied before clipping. Greedy selection and trial increments
remain colony-owned.

References
----------
Zhu, G., & Kwong, S. (2010). Gbest-guided artificial bee colony algorithm for
numerical function optimization. *Applied Mathematics and Computation*,
217(7), 3166–3173. https://doi.org/10.1016/j.amc.2010.08.049
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import SwarmSeekError
from swarm_seek.types import FloatArray
from swarm_seek.variants._helpers import (
    abandon_exhausted,
    best_index,
    partner_indices,
    require_population,
    require_sense,
    selection_probabilities,
    sparse_weights,
)
from swarm_seek.variants.registry import register_variant

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space

DEFAULT_C = 1.5


def _require_c(params: dict[str, float | int | str]) -> float:
    raw = params.get("C", DEFAULT_C)
    try:
        c_val = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"params['C'] must be a positive float; got {raw!r}."
        raise SwarmSeekError(msg) from exc
    if not np.isfinite(c_val) or c_val <= 0.0:
        msg = f"params['C'] must be a positive float; got {c_val}."
        raise SwarmSeekError(msg)
    return c_val


def _gabc_candidates(
    population: FloatArray,
    fitness: FloatArray,
    rng: np.random.Generator,
    *,
    space: Space,
    sense: str,
    c_val: float,
    source_indices: NDArray[np.integer] | None = None,
) -> FloatArray:
    """Propose GABC neighbors for all rows or for selected source indices."""
    pop = require_population(population, label="GABC")
    n_pop, n_dim = pop.shape
    fit = np.asarray(fitness, dtype=np.float64)
    if fit.shape != (n_pop,):
        msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
        raise SwarmSeekError(msg)

    gbest = pop[best_index(fit, sense=sense)]
    candidates = pop.copy()

    if source_indices is None:
        rows = np.arange(n_pop, dtype=np.int64)
    else:
        rows = np.asarray(source_indices, dtype=np.int64)

    # Unique last-write-wins order preserves onlooker overwrite semantics.
    for idx in rows:
        i = int(idx)
        partner = int(rng.integers(0, n_pop - 1))
        if partner >= i:
            partner += 1
        dim = int(rng.integers(0, n_dim))
        phi = float(rng.uniform(-1.0, 1.0))
        psi = float(rng.uniform(0.0, c_val))
        x = pop[i]
        xk = pop[partner]
        v = x.copy()
        v[dim] = x[dim] + phi * (x[dim] - xk[dim]) + psi * (gbest[dim] - x[dim])
        candidates[i] = space.repair(v)
    return candidates


def _gabc_batch(
    population: FloatArray,
    fitness: FloatArray,
    rng: np.random.Generator,
    *,
    space: Space,
    sense: str,
    c_val: float,
) -> FloatArray:
    """Vectorized employed-style GABC batch (one proposal per source)."""
    pop = require_population(population, label="GABC")
    n_pop, n_dim = pop.shape
    fit = np.asarray(fitness, dtype=np.float64)
    if fit.shape != (n_pop,):
        msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
        raise SwarmSeekError(msg)

    partners = partner_indices(n_pop, rng)
    phi, dims = sparse_weights(n_pop, n_dim, rng, low=-1.0, high=1.0)
    psi, _ = sparse_weights(n_pop, n_dim, rng, low=0.0, high=c_val, dims=dims)
    gbest = pop[best_index(fit, sense=sense)]
    raw = pop + phi * (pop - pop[partners]) + psi * (gbest - pop)
    return space.repair(raw)


class GbestABC:
    """Gbest-guided ABC employed / onlooker / scout strategy."""

    name = "gabc"
    citation = (
        "Zhu & Kwong (2010) Appl. Math. Comput. 217:3166–3173, "
        "doi:10.1016/j.amc.2010.08.049"
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
        """Propose GABC neighbors for every food source."""
        del backend
        sense = require_sense(params)
        return _gabc_batch(
            population,
            fitness,
            rng,
            space=space,
            sense=sense,
            c_val=_require_c(params),
        )

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
        """Propose GABC neighbors via fitness-proportional source selection."""
        del backend
        pop = require_population(population, label="GABC")
        n_pop = pop.shape[0]
        fit = np.asarray(fitness, dtype=np.float64)
        if fit.shape != (n_pop,):
            msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
            raise SwarmSeekError(msg)

        sense = require_sense(params)
        c_val = _require_c(params)
        probs = selection_probabilities(fit, sense=sense)
        selected = rng.choice(n_pop, size=n_pop, replace=True, p=probs)
        return _gabc_candidates(
            pop,
            fit,
            rng,
            space=space,
            sense=sense,
            c_val=c_val,
            source_indices=selected,
        )

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


register_variant("gabc", GbestABC, replace=True)
