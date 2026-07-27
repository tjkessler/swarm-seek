"""Synthetic TSP helpers for combinatorial ABC smokes."""

from __future__ import annotations

import numpy as np

from swarm_seek.types import FloatArray


def random_cities(
    n: int,
    *,
    seed: int | None = 0,
    low: float = 0.0,
    high: float = 1.0,
) -> FloatArray:
    """Return ``(n, 2)`` city coordinates in a box."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2)).astype(np.float64)


def tour_length(tours: FloatArray, cities: FloatArray) -> FloatArray:
    """Closed-tour Euclidean length for one or more permutations.

    Parameters
    ----------
    tours
        Shape ``(n_dim,)`` or ``(n, n_dim)`` with city index codes.
    cities
        Shape ``(n_dim, 2)`` coordinates.
    """
    coords = np.asarray(cities, dtype=np.float64)
    arr = np.asarray(tours, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    idx = np.rint(arr).astype(np.int64)
    lengths = np.empty(arr.shape[0], dtype=np.float64)
    for i, tour in enumerate(idx):
        pts = coords[tour]
        deltas = np.diff(pts, axis=0)
        dist = float(np.sum(np.sqrt(np.sum(deltas**2, axis=1))))
        dist += float(np.linalg.norm(pts[-1] - pts[0]))
        lengths[i] = dist
    return lengths[0] if single else lengths
