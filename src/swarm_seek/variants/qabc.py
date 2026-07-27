"""Quick Artificial Bee Colony (qABC) update rules.

Implements the onlooker neighborhood search of Karaboga & Gorkemli (2014).
Employed and scout phases match Original ABC. Onlookers roulette-select a food
source region, identify the best solution in that source's Euclidean
neighborhood, and search from that neighborhood best:

``v_{ij} = x^{N_m}_{\\mathrm{best},j} + \\phi_{ij}
(x^{N_m}_{\\mathrm{best},j} - x_{kj})``

with a single random dimension ``j`` and ``\\phi_{ij} \\in [-1, 1]``.

Neighborhood of food source ``m`` (Karaboga & Gorkemli, 2014, Eqs. 6–8):

``N_m = \\{ j : d(m, j) \\le r\\, md_m \\}``

where ``md_m`` is the mean Euclidean distance from ``x_m`` to the other
``SN - 1`` food sources. Source ``m`` is always included. When ``r = 0``,
the neighborhood is ``{m}``.

Parameter mapping
-----------------
limit
    Abandonment limit (shared with Original ABC).
sense
    ``\"minimize\"`` (default) or ``\"maximize\"``.
r
    Neighborhood radius factor (paper ``r``); default ``1.0``.

Notes
-----
Greedy selection and trial increments remain colony-owned. Onlookers
roulette-select a food-source region, then generate a candidate around that
region's neighborhood best; the candidate is applied to the neighborhood-best
index (Karaboga & Gorkemli, 2014: improve ``x^{N_m}_{best}``).

References
----------
Karaboga, D., & Gorkemli, B. (2014). A quick artificial bee colony (qABC)
algorithm and its performance on optimization problems. *Applied Soft
Computing*, 23, 227–238. https://doi.org/10.1016/j.asoc.2014.06.035
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
    require_population,
    require_sense,
    selection_probabilities,
)
from swarm_seek.variants.original import _neighbor_candidates
from swarm_seek.variants.registry import register_variant

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space

DEFAULT_R = 1.0


def _require_r(params: dict[str, float | int | str]) -> float:
    raw = params.get("r", DEFAULT_R)
    try:
        r_val = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"params['r'] must be a non-negative float; got {raw!r}."
        raise SwarmSeekError(msg) from exc
    if not np.isfinite(r_val) or r_val < 0.0:
        msg = f"params['r'] must be a non-negative float; got {r_val}."
        raise SwarmSeekError(msg)
    return r_val


def mean_pairwise_distance(population: FloatArray) -> float:
    """Return mean Euclidean distance over unique food-source pairs.

    Prefer :func:`mean_distance_from_source` for qABC neighborhoods (paper
    Eq. 6 uses a per-source mean).
    """
    pop = np.asarray(population, dtype=np.float64)
    n_pop = pop.shape[0]
    if n_pop < 2:
        return 0.0
    diff = pop[:, None, :] - pop[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n_pop, k=1)
    return float(np.mean(dists[iu]))


def mean_distance_from_source(population: FloatArray, source_idx: int) -> float:
    """Return mean Euclidean distance from ``source_idx`` to other sources.

    Implements Karaboga & Gorkemli (2014) Eq. (6): ``md_m``.
    """
    pop = np.asarray(population, dtype=np.float64)
    n_pop = pop.shape[0]
    idx = int(source_idx)
    if n_pop < 2:
        return 0.0
    if idx < 0 or idx >= n_pop:
        msg = f"source_idx out of range for population size {n_pop}: {idx}."
        raise SwarmSeekError(msg)
    dists = np.linalg.norm(pop - pop[idx], axis=1)
    return float(np.sum(dists) / (n_pop - 1))


def neighborhood_indices(
    population: FloatArray,
    source_idx: int,
    *,
    r: float,
    d_md: float | None = None,
) -> NDArray[np.int64]:
    """Return indices in the Euclidean neighborhood of ``source_idx``.

    Always includes ``source_idx``. When ``r == 0`` or ``md_m == 0``, the
    neighborhood is only ``{source_idx}``. ``d_md`` overrides Eq. (6) when set.
    """
    pop = np.asarray(population, dtype=np.float64)
    n_pop = pop.shape[0]
    idx = int(source_idx)
    if idx < 0 or idx >= n_pop:
        msg = f"source_idx out of range for population size {n_pop}: {idx}."
        raise SwarmSeekError(msg)

    if r == 0.0:
        return np.array([idx], dtype=np.int64)

    mean_dist = mean_distance_from_source(pop, idx) if d_md is None else float(d_md)
    if mean_dist <= 0.0:
        return np.array([idx], dtype=np.int64)

    dists = np.linalg.norm(pop - pop[idx], axis=1)
    members = np.nonzero(dists <= r * mean_dist)[0].astype(np.int64)
    if idx not in members:
        members = np.append(members, np.int64(idx))
    return members


def neighborhood_best_index(
    population: FloatArray,
    fitness: FloatArray,
    source_idx: int,
    *,
    r: float,
    sense: str,
    d_md: float | None = None,
) -> int:
    """Return the best food-source index in the neighborhood of ``source_idx``."""
    members = neighborhood_indices(population, source_idx, r=r, d_md=d_md)
    fit = np.asarray(fitness, dtype=np.float64)
    local = fit[members]
    local_best = best_index(local, sense=sense)
    return int(members[local_best])


class QuickABC:
    """Quick ABC employed / onlooker / scout strategy."""

    name = "qabc"
    citation = (
        "Karaboga & Gorkemli (2014) Appl. Soft Comput. 23:227–238, "
        "doi:10.1016/j.asoc.2014.06.035"
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
        """Propose Original-ABC neighbors for every food source."""
        del fitness, params
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
        """Propose onlooker candidates from each selected source's neighborhood best."""
        del backend
        pop = require_population(population, label="qABC")
        n_pop, n_dim = pop.shape
        fit = np.asarray(fitness, dtype=np.float64)
        if fit.shape != (n_pop,):
            msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
            raise SwarmSeekError(msg)

        sense = require_sense(params)
        r_val = _require_r(params)
        probs = selection_probabilities(fit, sense=sense)
        selected = rng.choice(n_pop, size=n_pop, replace=True, p=probs)

        candidates = pop.copy()
        for source_idx in selected:
            region = int(source_idx)
            best_n = neighborhood_best_index(pop, fit, region, r=r_val, sense=sense)
            partner = int(rng.integers(0, n_pop - 1))
            if partner >= best_n:
                partner += 1

            dim = int(rng.integers(0, n_dim))
            phi = float(rng.uniform(-1.0, 1.0))
            base = pop[best_n]
            v = base.copy()
            v[dim] = base[dim] + phi * (base[dim] - pop[partner, dim])
            # Improve the neighborhood best (not only the region centre).
            candidates[best_n] = space.repair(v)
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


register_variant("qabc", QuickABC, replace=True)
