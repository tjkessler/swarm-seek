"""Combinatorial Artificial Bee Colony (CABC) for permutation spaces.

Implements the combinatorial ABC approach of Karaboga & Gorkemli (2011):
the continuous neighbor-update equation of basic ABC is replaced by
**discrete neighborhood operators** on tour permutations (TSP-style
encoding). Swarm Seek provides **swap** and **insertion** operators on
:class:`~swarm_seek.space.PermutationSpace` (design Q8).

The employed / onlooker / scout phase structure matches ABC; greedy
selection and trial counters remain colony-owned. Continuous
``BackendProtocol.generate_candidates`` is unused (backends are ignored).

Operator locators
-----------------
swap
    Exchange two distinct positions in the tour (GA swap mutation).
insertion
    Remove the city at position ``i`` and insert it at position ``j``.

Parameter mapping
-----------------
limit
    Abandonment limit (shared ABC parameter).
operator
    ``\"swap\"``, ``\"insertion\"``, or ``\"random\"`` (default: choose
    uniformly between swap and insertion each proposal).
sense
    ``\"minimize\"`` or ``\"maximize\"``.

References
----------
Karaboga, D., & Gorkemli, B. (2011). A combinatorial Artificial Bee Colony
algorithm for traveling salesman problem. *INISTA 2011*, pp. 50–53.
https://doi.org/10.1109/INISTA.2011.5946125
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import SwarmSeekError
from swarm_seek.types import FloatArray
from swarm_seek.variants._helpers import (
    abandon_exhausted,
    require_population,
    require_sense,
    selection_probabilities,
)
from swarm_seek.variants.registry import register_variant

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space


def _swap_tour(tour: FloatArray, rng: np.random.Generator) -> FloatArray:
    out = np.asarray(tour, dtype=np.float64).copy()
    n = out.shape[0]
    i, j = rng.choice(n, size=2, replace=False)
    out[i], out[j] = out[j], out[i]
    return out


def _insertion_tour(tour: FloatArray, rng: np.random.Generator) -> FloatArray:
    out = np.asarray(tour, dtype=np.float64).copy()
    n = out.shape[0]
    i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
    city = out[i]
    out = np.delete(out, i)
    # After delete, insertion index shifts if j > i.
    insert_at = j if j < i else j - 1
    insert_at = int(np.clip(insert_at, 0, n - 1))
    out = np.insert(out, insert_at, city)
    return out.astype(np.float64, copy=False)


def _apply_operator(
    tour: FloatArray,
    rng: np.random.Generator,
    operator: str,
) -> FloatArray:
    if operator == "random":
        operator = "swap" if rng.random() < 0.5 else "insertion"
    if operator == "swap":
        return _swap_tour(tour, rng)
    if operator == "insertion":
        return _insertion_tour(tour, rng)
    msg = (
        "params['operator'] must be 'swap', 'insertion', or 'random'; "
        f"got {operator!r}."
    )
    raise SwarmSeekError(msg)


def _require_operator(params: dict[str, float | int | str]) -> str:
    op = str(params.get("operator", "random"))
    if op not in {"swap", "insertion", "random"}:
        msg = (
            f"params['operator'] must be 'swap', 'insertion', or 'random'; got {op!r}."
        )
        raise SwarmSeekError(msg)
    return op


def _neighbor_row(
    population: FloatArray,
    index: int,
    rng: np.random.Generator,
    *,
    space: Space,
    operator: str,
) -> FloatArray:
    tour = population[index]
    candidate = _apply_operator(tour, rng, operator)
    repaired = space.repair(candidate)
    return np.asarray(repaired, dtype=np.float64).reshape(-1)


class CombinatorialABC:
    """Combinatorial ABC (CABC) employed / onlooker / scout strategy."""

    name = "cabc"
    citation = (
        "Karaboga & Gorkemli (2011) INISTA, pp. 50–53, doi:10.1109/INISTA.2011.5946125"
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
        """Propose one discrete neighbor per food source."""
        del fitness, backend
        pop = require_population(population, label="CABC")
        operator = _require_operator(params)
        out = np.empty_like(pop, dtype=np.float64)
        for i in range(pop.shape[0]):
            out[i] = _neighbor_row(pop, i, rng, space=space, operator=operator)
        return out

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
        """Roulette-select sources and propose discrete neighbors."""
        del backend
        pop = require_population(population, label="CABC")
        n_pop = pop.shape[0]
        fit = np.asarray(fitness, dtype=np.float64)
        if fit.shape != (n_pop,):
            msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
            raise SwarmSeekError(msg)
        sense = require_sense(params)
        operator = _require_operator(params)
        probs = selection_probabilities(fit, sense=sense)
        selected = rng.choice(n_pop, size=n_pop, replace=True, p=probs)

        candidates = pop.copy()
        for source_idx in selected:
            idx = int(source_idx)
            candidates[idx] = _neighbor_row(
                pop, idx, rng, space=space, operator=operator
            )
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
        """Reinitialize abandoned tours via ``space.sample``."""
        del fitness, backend
        return abandon_exhausted(population, trials, rng, space=space, params=params)


register_variant("cabc", CombinatorialABC, replace=True)
