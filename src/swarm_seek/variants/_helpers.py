"""Shared helpers for continuous ABC variant strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from swarm_seek.errors import SwarmSeekError
from swarm_seek.types import FloatArray
from swarm_seek.variants.params import require_limit

if TYPE_CHECKING:
    from swarm_seek.space import Space


def require_population(population: FloatArray, *, label: str = "ABC") -> FloatArray:
    """Validate and return a float64 population with ``n >= 2``."""
    pop = np.asarray(population, dtype=np.float64)
    if pop.ndim != 2 or pop.shape[0] < 2:
        msg = f"{label} requires population shape (n, d) with n >= 2; got {pop.shape}."
        raise SwarmSeekError(msg)
    return pop


def require_sense(params: dict[str, float | int | str]) -> str:
    """Return ``minimize`` or ``maximize`` from ``params``."""
    sense = str(params.get("sense", "minimize"))
    if sense not in {"minimize", "maximize"}:
        msg = f"params['sense'] must be 'minimize' or 'maximize'; got {sense!r}."
        raise SwarmSeekError(msg)
    return sense


def partner_indices(n_pop: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Sample partner index ``k ≠ i`` for each food source ``i``."""
    partners = rng.integers(0, n_pop - 1, size=n_pop, dtype=np.int64)
    return partners + (partners >= np.arange(n_pop, dtype=np.int64))


def sparse_phi(
    n_pop: int,
    n_dim: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Build ``(n, d)`` φ with one uniform ``[-1, 1]`` entry per row."""
    dims = rng.integers(0, n_dim, size=n_pop)
    phi = np.zeros((n_pop, n_dim), dtype=np.float64)
    phi[np.arange(n_pop), dims] = rng.uniform(-1.0, 1.0, size=n_pop)
    return phi


def sparse_weights(
    n_pop: int,
    n_dim: int,
    rng: np.random.Generator,
    *,
    low: float,
    high: float,
    dims: NDArray[np.integer] | None = None,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Build sparse weights on one dimension per row.

    Returns
    -------
    weights, dims
        ``weights`` has shape ``(n, d)``; ``dims`` has shape ``(n,)``.
    """
    if dims is None:
        chosen = rng.integers(0, n_dim, size=n_pop).astype(np.int64)
    else:
        chosen = np.asarray(dims, dtype=np.int64)
    weights = np.zeros((n_pop, n_dim), dtype=np.float64)
    weights[np.arange(n_pop), chosen] = rng.uniform(low, high, size=n_pop)
    return weights, chosen


def nectar(fitness: FloatArray, *, sense: str) -> FloatArray:
    """Convert objective values to non-negative nectar amounts for roulette."""
    values = np.asarray(fitness, dtype=np.float64)
    if sense == "maximize":
        shifted = values - np.min(values)
        return shifted + 1e-12
    # Karaboga nectar transform for minimization (Karaboga & Basturk, 2007).
    out = np.empty_like(values, dtype=np.float64)
    nonneg = values >= 0.0
    out[nonneg] = 1.0 / (1.0 + values[nonneg])
    out[~nonneg] = 1.0 + np.abs(values[~nonneg])
    return out


def selection_probabilities(fitness: FloatArray, *, sense: str) -> FloatArray:
    """Return roulette probabilities from objective fitness."""
    amounts = nectar(fitness, sense=sense)
    total = float(np.sum(amounts))
    if not np.isfinite(total) or total <= 0.0:
        n = amounts.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    return amounts / total


def best_index(fitness: FloatArray, *, sense: str) -> int:
    """Return the index of the current global-best food source."""
    values = np.asarray(fitness, dtype=np.float64)
    if sense == "maximize":
        return int(np.argmax(values))
    return int(np.argmin(values))


def abandon_exhausted(
    population: FloatArray,
    trials: NDArray[np.integer],
    rng: np.random.Generator,
    *,
    space: Space,
    params: dict[str, float | int | str],
) -> tuple[FloatArray, NDArray[np.integer]]:
    """Reinitialize sources with ``trials >= limit`` and reset their counters."""
    pop = np.asarray(population, dtype=np.float64).copy()
    trial_counts = np.asarray(trials).copy()
    if pop.ndim != 2:
        msg = f"population must be 2-D; got shape {pop.shape}."
        raise SwarmSeekError(msg)
    if trial_counts.shape != (pop.shape[0],):
        msg = f"trials must have shape ({pop.shape[0]},); got {trial_counts.shape}."
        raise SwarmSeekError(msg)

    limit = require_limit(params)
    abandoned = np.nonzero(trial_counts >= limit)[0]
    if abandoned.size:
        pop[abandoned] = space.sample(int(abandoned.size), rng)
        trial_counts[abandoned] = 0
    return pop, trial_counts
