#!/usr/bin/env python3
"""Compare continuous ``generate_candidates`` wall time across backends.

Not a CI wall-clock gate. Install optional extras for Numba/JAX timings::

    pip install 'swarm-seek[numba,jax]'
    python scripts/compare_backends.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from swarm_seek.backends import get_backend
from swarm_seek.space import ContinuousSpace


def _time_backend(name: str, *, n_pop: int, n_dim: int, repeats: int) -> float | None:
    try:
        backend = get_backend(name)
    except (ImportError, Exception) as exc:  # noqa: BLE001 — report and skip
        print(f"{name:8s}  unavailable ({exc})")
        return None
    space = ContinuousSpace([(-5.0, 5.0)] * n_dim)
    rng = np.random.default_rng(0)
    pop = space.sample(n_pop, rng)
    partners = rng.integers(0, n_pop, size=n_pop)
    partners = partners + (partners >= np.arange(n_pop))
    partners = partners % n_pop
    # Ensure partner != i
    partners = np.array(
        [(p if p != i else (p + 1) % n_pop) for i, p in enumerate(partners)],
        dtype=np.int64,
    )
    phi = np.zeros((n_pop, n_dim), dtype=np.float64)
    dims = rng.integers(0, n_dim, size=n_pop)
    phi[np.arange(n_pop), dims] = rng.uniform(-1.0, 1.0, size=n_pop)

    # Warmup (esp. Numba JIT / JAX compile).
    backend.generate_candidates(pop, partners, phi, space=space)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        backend.generate_candidates(pop, partners, phi, space=space)
        times.append(time.perf_counter() - t0)
    mean_s = float(np.mean(times))
    print(f"{backend.name:8s}  mean={mean_s * 1e3:8.3f} ms  (n={n_pop}, d={n_dim})")
    return mean_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pop", type=int, default=200)
    parser.add_argument("--n-dim", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print("Backend generate_candidates timing (lower is faster)")
    for name in ("numpy", "numba", "jax"):
        _time_backend(name, n_pop=args.n_pop, n_dim=args.n_dim, repeats=args.repeats)


if __name__ == "__main__":
    main()
