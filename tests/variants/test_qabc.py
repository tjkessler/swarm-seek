"""Tests for Quick ABC update rules."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend
from swarm_seek.errors import SwarmSeekError
from swarm_seek.space import ContinuousSpace
from swarm_seek.variants import get_variant, register_variant
from swarm_seek.variants.qabc import (
    QuickABC,
    _require_r,
    mean_distance_from_source,
    mean_pairwise_distance,
    neighborhood_best_index,
    neighborhood_indices,
)


@pytest.fixture(autouse=True)
def _ensure_qabc_registered() -> None:
    register_variant("qabc", QuickABC, replace=True)
    yield


def test_get_variant_qabc() -> None:
    strategy = get_variant("qabc")
    assert isinstance(strategy, QuickABC)
    assert strategy.name == "qabc"
    assert "Gorkemli" in strategy.citation


def test_mean_pairwise_distance() -> None:
    pop = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    # pairs: (0,1)=5, (0,2)=0, (1,2)=5 → mean 10/3
    np.testing.assert_allclose(mean_pairwise_distance(pop), 10.0 / 3.0)


def test_mean_distance_from_source_eq6() -> None:
    # Points on a line at 0, 1, 10. For source 0: (1+10)/2 = 5.5
    pop = np.array([[0.0], [1.0], [10.0]], dtype=np.float64)
    np.testing.assert_allclose(mean_distance_from_source(pop, 0), 5.5)


def test_neighborhood_r_zero_is_self_only() -> None:
    pop = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    members = neighborhood_indices(pop, 1, r=0.0)
    np.testing.assert_array_equal(members, [1])


def test_neighborhood_membership_with_radius() -> None:
    # Points on a line at 0, 1, 10. Per-source md_0 = 5.5; r=0.2 → radius 1.1.
    pop = np.array([[0.0], [1.0], [10.0]], dtype=np.float64)
    members = neighborhood_indices(pop, 0, r=0.2)
    assert 0 in members
    assert 1 in members
    assert 2 not in members


def test_neighborhood_best_prefers_better_neighbor() -> None:
    pop = np.array([[0.0], [1.0], [10.0]], dtype=np.float64)
    fitness = np.array([5.0, 0.0, 1.0])  # source 1 best overall
    best = neighborhood_best_index(pop, fitness, 0, r=0.2, sense="minimize")
    assert best == 1


def test_onlooker_uses_neighborhood_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ContinuousSpace([(-20.0, 20.0)])
    strategy = QuickABC()
    # Source 0 selected; neighborhood best is source 1 at 2.0.
    population = np.array([[0.0], [2.0], [10.0]], dtype=np.float64)
    fitness = np.array([10.0, 0.0, 5.0])

    class _Rng:
        def choice(self, n_pop, size=None, replace=True, p=None):
            return np.zeros(int(size), dtype=np.int64)  # always select source 0

        def integers(self, low, high=None, size=None, dtype=None):
            # partner: high == n_pop - 1 == 2 → return 1 → partner 2 when best_n=1
            # dim: high == n_dim == 1 → return 0
            if high == 2:
                return 1
            return 0

        def uniform(self, low, high, size=None):
            return 0.5

    import swarm_seek.variants.qabc as qabc_mod

    monkeypatch.setattr(
        qabc_mod,
        "neighborhood_best_index",
        lambda *args, **kwargs: 1,
    )

    out = strategy.onlooker_step(
        population,
        fitness,
        _Rng(),  # type: ignore[arg-type]
        backend=NumpyBackend(),
        space=space,
        params={"r": 1.0, "sense": "minimize"},
    )
    # v = 2 + 0.5*(2 - 10) = -2, written to neighborhood best index 1
    np.testing.assert_array_equal(out[0], population[0])
    np.testing.assert_allclose(out[1], [-2.0], atol=1e-12)
    np.testing.assert_array_equal(out[2], population[2])


def test_employed_deterministic_under_seed() -> None:
    strategy = QuickABC()
    space = ContinuousSpace([(-5.0, 5.0)] * 2)
    population = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    fitness = np.zeros(3)
    kwargs = dict(backend=NumpyBackend(), space=space, params={})
    a = strategy.employed_step(population, fitness, np.random.default_rng(9), **kwargs)
    b = strategy.employed_step(population, fitness, np.random.default_rng(9), **kwargs)
    np.testing.assert_allclose(a, b)


def test_scout_abandons() -> None:
    strategy = QuickABC()
    space = ContinuousSpace([(0.0, 1.0)])
    population = np.array([[0.2], [0.8]], dtype=np.float64)
    trials = np.array([1, 4], dtype=np.int64)
    _, new_trials = strategy.scout_step(
        population,
        np.zeros(2),
        trials,
        np.random.default_rng(0),
        backend=NumpyBackend(),
        space=space,
        params={"limit": 4},
    )
    assert new_trials[1] == 0
    assert new_trials[0] == 1


def test_require_r_validation() -> None:
    assert _require_r({}) == 1.0
    assert _require_r({"r": 0}) == 0.0
    with pytest.raises(SwarmSeekError, match="r"):
        _require_r({"r": -1})
    with pytest.raises(SwarmSeekError, match="r"):
        _require_r({"r": "nope"})


def test_mean_pairwise_distance_singleton_is_zero() -> None:
    pop = np.array([[1.0, 2.0]], dtype=np.float64)
    assert mean_pairwise_distance(pop) == 0.0


def test_mean_distance_from_source_edge_cases() -> None:
    singleton = np.array([[1.0]], dtype=np.float64)
    assert mean_distance_from_source(singleton, 0) == 0.0
    pop = np.array([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(SwarmSeekError, match="source_idx"):
        mean_distance_from_source(pop, -1)
    with pytest.raises(SwarmSeekError, match="source_idx"):
        mean_distance_from_source(pop, 2)


def test_neighborhood_indices_validation_and_zero_md() -> None:
    pop = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    with pytest.raises(SwarmSeekError, match="source_idx"):
        neighborhood_indices(pop, 99, r=1.0)
    # Explicit zero mean distance → self-only neighborhood
    members = neighborhood_indices(pop, 1, r=1.0, d_md=0.0)
    np.testing.assert_array_equal(members, [1])


def test_onlooker_rejects_bad_fitness_shape() -> None:
    strategy = QuickABC()
    space = ContinuousSpace([(-5.0, 5.0)])
    population = np.array([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(SwarmSeekError, match="fitness"):
        strategy.onlooker_step(
            population,
            np.zeros(3),
            np.random.default_rng(0),
            backend=NumpyBackend(),
            space=space,
            params={"r": 1.0, "sense": "minimize"},
        )
