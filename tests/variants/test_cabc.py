"""Tests for combinatorial ABC (CABC)."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek import ABC, PermutationSpace
from swarm_seek.backends import NumpyBackend
from swarm_seek.benchmarks import random_cities, tour_length
from swarm_seek.errors import SwarmSeekError
from swarm_seek.variants import get_variant


def test_cabc_registered() -> None:
    strategy = get_variant("cabc")
    assert strategy.name == "cabc"


def test_cabc_operators_and_bad_operator() -> None:
    space = PermutationSpace(6)
    rng = np.random.default_rng(0)
    pop = space.sample(4, rng)
    fit = np.arange(4, dtype=np.float64)
    strategy = get_variant("cabc")
    backend = NumpyBackend()
    for operator in ("swap", "insertion", "random"):
        cand = strategy.employed_step(
            pop,
            fit,
            rng,
            backend=backend,
            space=space,
            params={"limit": 10, "sense": "minimize", "operator": operator},
        )
        assert cand.shape == pop.shape
        for row in cand:
            space.validate(row)
    with pytest.raises(SwarmSeekError, match="operator"):
        strategy.employed_step(
            pop,
            fit,
            rng,
            backend=backend,
            space=space,
            params={"limit": 10, "sense": "minimize", "operator": "2opt"},
        )


def test_cabc_tsp_smoke_beats_random() -> None:
    """Synthetic TSP smoke (design Q12): improve vs mean random tour."""
    n_cities = 12
    cities = random_cities(n_cities, seed=0)
    space = PermutationSpace(n_cities)
    rng = np.random.default_rng(1)
    random_tours = space.sample(200, rng)
    random_mean = float(np.mean(tour_length(random_tours, cities)))

    abc = ABC(
        space,
        variant="cabc",
        pop_size=16,
        limit=40,
        max_evals=2_000,
        seed=0,
        operator="random",
    )
    best = abc.minimize(lambda x: tour_length(x, cities))
    assert best.fitness < random_mean
    space.validate(best.x)
