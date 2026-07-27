"""Modified Artificial Bee Colony (MABC) update rules.

Implements the modification-rate / scaling search of Akay & Karaboga (2012).
Relative to Original ABC (single random dimension), each coordinate ``j`` of a
food source is perturbed independently with probability ``MR``:

``v_{ij} = x_{ij} + \\phi_{ij} (x_{ij} - x_{kj})``

with ``\\phi_{ij} \\in [-SF, SF]``. If no coordinate is selected by the
Bernoulli draws, one dimension is forced so every proposal changes at least
one coordinate (Akay & Karaboga, 2012, modified search equation / control
parameters ``MR`` and scaling factor).

Parameter mapping
-----------------
limit
    Abandonment limit (shared with Original ABC).
sense
    ``\"minimize\"`` (default) or ``\"maximize\"``.
MR
    Modification rate in ``[0, 1]``; default ``0.4``.
SF
    Scaling factor for ``\\phi`` magnitude; default ``1.0``.

Notes
-----
One partner ``k ≠ i`` is drawn per food source and shared across mutated
dimensions. Bound handling uses ``space.repair`` after the full update.
Greedy selection and trial increments remain colony-owned.

References
----------
Akay, B., & Karaboga, D. (2012). A modified artificial bee colony algorithm
for real-parameter optimization. *Information Sciences*, 192, 120–142.
https://doi.org/10.1016/j.ins.2010.07.015
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
)
from swarm_seek.variants.registry import register_variant

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space

DEFAULT_MR = 0.4
DEFAULT_SF = 1.0


def _require_mr(params: dict[str, float | int | str]) -> float:
    raw = params.get("MR", DEFAULT_MR)
    try:
        mr = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"params['MR'] must be a float in [0, 1]; got {raw!r}."
        raise SwarmSeekError(msg) from exc
    if not np.isfinite(mr) or mr < 0.0 or mr > 1.0:
        msg = f"params['MR'] must be a float in [0, 1]; got {mr}."
        raise SwarmSeekError(msg)
    return mr


def _require_sf(params: dict[str, float | int | str]) -> float:
    raw = params.get("SF", DEFAULT_SF)
    try:
        sf = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"params['SF'] must be a positive float; got {raw!r}."
        raise SwarmSeekError(msg) from exc
    if not np.isfinite(sf) or sf <= 0.0:
        msg = f"params['SF'] must be a positive float; got {sf}."
        raise SwarmSeekError(msg)
    return sf


def modification_mask(
    n_pop: int,
    n_dim: int,
    rng: np.random.Generator,
    *,
    mr: float,
) -> NDArray[np.bool_]:
    """Return a boolean mask of shape ``(n, d)`` with ≥1 True per row."""
    mask = rng.random((n_pop, n_dim)) < mr
    empty_rows = ~np.any(mask, axis=1)
    if np.any(empty_rows):
        dims = rng.integers(0, n_dim, size=int(np.count_nonzero(empty_rows)))
        mask[np.nonzero(empty_rows)[0], dims] = True
    return mask


def _mabc_phi(
    mask: NDArray[np.bool_],
    rng: np.random.Generator,
    *,
    sf: float,
) -> FloatArray:
    phi = np.zeros(mask.shape, dtype=np.float64)
    phi[mask] = rng.uniform(-sf, sf, size=int(np.count_nonzero(mask)))
    return phi


def _mabc_batch(
    population: FloatArray,
    rng: np.random.Generator,
    *,
    backend: BackendProtocol,
    space: Space,
    mr: float,
    sf: float,
) -> FloatArray:
    pop = require_population(population, label="MABC")
    n_pop, n_dim = pop.shape
    partners = partner_indices(n_pop, rng)
    mask = modification_mask(n_pop, n_dim, rng, mr=mr)
    phi = _mabc_phi(mask, rng, sf=sf)
    return backend.generate_candidates(pop, partners, phi, space=space)


class ModifiedABC:
    """Modified ABC employed / onlooker / scout strategy."""

    name = "mabc"
    citation = (
        "Akay & Karaboga (2012) Inf. Sci. 192:120–142, doi:10.1016/j.ins.2010.07.015"
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
        """Propose MABC neighbors for every food source."""
        del fitness
        return _mabc_batch(
            population,
            rng,
            backend=backend,
            space=space,
            mr=_require_mr(params),
            sf=_require_sf(params),
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
        """Propose MABC neighbors via fitness-proportional source selection."""
        pop = require_population(population, label="MABC")
        n_pop, n_dim = pop.shape
        fit = np.asarray(fitness, dtype=np.float64)
        if fit.shape != (n_pop,):
            msg = f"fitness must have shape ({n_pop},); got {fit.shape}."
            raise SwarmSeekError(msg)

        sense = require_sense(params)
        mr = _require_mr(params)
        sf = _require_sf(params)
        probs = selection_probabilities(fit, sense=sense)
        selected = rng.choice(n_pop, size=n_pop, replace=True, p=probs)

        candidates = pop.copy()
        for source_idx in selected:
            idx = int(source_idx)
            partner = int(rng.integers(0, n_pop - 1))
            if partner >= idx:
                partner += 1
            mask = modification_mask(1, n_dim, rng, mr=mr)
            phi = np.zeros((n_pop, n_dim), dtype=np.float64)
            phi[idx] = _mabc_phi(mask, rng, sf=sf)[0]
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


register_variant("mabc", ModifiedABC, replace=True)
