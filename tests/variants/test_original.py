"""Tests for Original ABC update rules."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend
from swarm_seek.errors import SwarmSeekError, UnknownVariantError
from swarm_seek.space import ContinuousSpace
from swarm_seek.variants import clear_registry, get_variant, register_variant
from swarm_seek.variants.original import (
    OriginalABC,
    _nectar,
    _partner_indices,
    _sparse_phi,
)


@pytest.fixture(autouse=True)
def _ensure_original_registered() -> None:
    register_variant("original", OriginalABC, replace=True)
    yield


def test_get_variant_original() -> None:
    strategy = get_variant("original")
    assert isinstance(strategy, OriginalABC)
    assert strategy.name == "original"
    assert "Karaboga" in strategy.citation


def test_partner_indices_never_self() -> None:
    rng = np.random.default_rng(0)
    partners = _partner_indices(5, rng)
    assert partners.shape == (5,)
    assert np.all(partners != np.arange(5))
    assert np.all((partners >= 0) & (partners < 5))


def test_sparse_phi_single_dimension_per_row() -> None:
    rng = np.random.default_rng(1)
    phi = _sparse_phi(4, 3, rng)
    assert phi.shape == (4, 3)
    assert np.all(np.count_nonzero(phi, axis=1) == 1)
    assert np.all((phi >= -1.0) & (phi <= 1.0))


def test_employed_neighbor_formula_hand_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ContinuousSpace([(-10.0, 10.0), (-10.0, 10.0)])
    backend = NumpyBackend()
    strategy = OriginalABC()
    population = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, -1.0]], dtype=np.float64)
    fitness = np.zeros(3)
    rng = np.random.default_rng(0)

    import swarm_seek.variants.original as original_mod

    monkeypatch.setattr(
        original_mod,
        "_partner_indices",
        lambda n_pop, _rng: np.array([1, 2, 0], dtype=np.int64),
    )
    monkeypatch.setattr(
        original_mod,
        "_sparse_phi",
        lambda n_pop, n_dim, _rng: np.array(
            [[0.5, 0.0], [0.0, -0.5], [1.0, 0.0]], dtype=np.float64
        ),
    )

    got = strategy.employed_step(
        population,
        fitness,
        rng,
        backend=backend,
        space=space,
        params={},
    )

    # v0 = [0,0] + [0.5,0]*([0,0]-[2,4]) = [-1, 0]
    # v1 = [2,4] + [0,-0.5]*([2,4]-[1,-1]) = [2, 1.5]
    # v2 = [1,-1] + [1,0]*([1,-1]-[0,0]) = [2, -1]
    expected = np.array([[-1.0, 0.0], [2.0, 1.5], [2.0, -1.0]], dtype=np.float64)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_array_equal(population, [[0.0, 0.0], [2.0, 4.0], [1.0, -1.0]])


def test_employed_is_deterministic_under_seed() -> None:
    space = ContinuousSpace([(-5.0, 5.0)] * 3)
    backend = NumpyBackend()
    strategy = OriginalABC()
    population = np.array(
        [[0.0, 1.0, -1.0], [2.0, 0.0, 0.5], [-1.0, -2.0, 1.5]],
        dtype=np.float64,
    )
    fitness = np.array([1.0, 2.0, 0.5])
    a = strategy.employed_step(
        population,
        fitness,
        np.random.default_rng(42),
        backend=backend,
        space=space,
        params={},
    )
    b = strategy.employed_step(
        population,
        fitness,
        np.random.default_rng(42),
        backend=backend,
        space=space,
        params={},
    )
    np.testing.assert_allclose(a, b)


def test_employed_repairs_out_of_bounds() -> None:
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    backend = NumpyBackend()
    strategy = OriginalABC()
    population = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    out = strategy.employed_step(
        population,
        np.zeros(2),
        np.random.default_rng(7),
        backend=backend,
        space=space,
        params={},
    )
    assert out.shape == (2, 2)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_nectar_minimize_transform() -> None:
    fit = np.array([0.0, 1.0, -2.0])
    nectar = _nectar(fit, sense="minimize")
    np.testing.assert_allclose(nectar, [1.0, 0.5, 3.0])


def test_nectar_maximize_shifts_to_nonnegative() -> None:
    nectar = _nectar(np.array([-1.0, 0.0, 3.0]), sense="maximize")
    assert np.all(nectar > 0.0)
    assert nectar[2] > nectar[1] > nectar[0]


def test_onlooker_rejects_bad_fitness_shape_and_sense() -> None:
    strategy = OriginalABC()
    population = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    backend = NumpyBackend()
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    with pytest.raises(SwarmSeekError, match="fitness must have shape"):
        strategy.onlooker_step(
            population,
            np.zeros(3),
            np.random.default_rng(0),
            backend=backend,
            space=space,
            params={},
        )
    with pytest.raises(SwarmSeekError, match="sense"):
        strategy.onlooker_step(
            population,
            np.zeros(2),
            np.random.default_rng(0),
            backend=backend,
            space=space,
            params={"sense": "sideways"},
        )


def test_scout_rejects_bad_shapes() -> None:
    strategy = OriginalABC()
    backend = NumpyBackend()
    space = ContinuousSpace([(0.0, 1.0)])
    with pytest.raises(SwarmSeekError, match="2-D"):
        strategy.scout_step(
            np.array([0.0, 1.0]),
            np.zeros(2),
            np.zeros(2, dtype=np.int64),
            np.random.default_rng(0),
            backend=backend,
            space=space,
            params={},
        )
    with pytest.raises(SwarmSeekError, match="trials must have shape"):
        strategy.scout_step(
            np.array([[0.0], [1.0]]),
            np.zeros(2),
            np.zeros(3, dtype=np.int64),
            np.random.default_rng(0),
            backend=backend,
            space=space,
            params={},
        )


def test_onlooker_prefers_better_sources() -> None:
    space = ContinuousSpace([(-1.0, 1.0), (-1.0, 1.0)])
    backend = NumpyBackend()
    strategy = OriginalABC()
    population = np.array(
        [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.25]],
        dtype=np.float64,
    )
    fitness = np.array([0.0, 100.0, 100.0])
    changed = np.zeros(3, dtype=int)
    for seed in range(200):
        out = strategy.onlooker_step(
            population,
            fitness,
            np.random.default_rng(seed),
            backend=backend,
            space=space,
            params={"sense": "minimize"},
        )
        changed += np.any(out != population, axis=1).astype(int)
    assert changed[0] > changed[1]
    assert changed[0] > changed[2]


def test_scout_abandons_and_resets_trials() -> None:
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    backend = NumpyBackend()
    strategy = OriginalABC()
    population = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    trials = np.array([0, 5, 2], dtype=np.int64)
    new_pop, new_trials = strategy.scout_step(
        population,
        np.zeros(3),
        trials,
        np.random.default_rng(0),
        backend=backend,
        space=space,
        params={"limit": 5},
    )
    assert new_trials[1] == 0
    assert new_trials[0] == 0
    assert new_trials[2] == 2
    assert np.all(new_pop[1] >= 0.0) and np.all(new_pop[1] <= 1.0)
    np.testing.assert_array_equal(new_pop[0], population[0])
    np.testing.assert_array_equal(new_pop[2], population[2])
    np.testing.assert_array_equal(trials, [0, 5, 2])


def test_scout_no_abandon_when_below_limit() -> None:
    space = ContinuousSpace([(0.0, 1.0)])
    strategy = OriginalABC()
    population = np.array([[0.2], [0.8]], dtype=np.float64)
    trials = np.array([1, 2], dtype=np.int64)
    new_pop, new_trials = strategy.scout_step(
        population,
        np.zeros(2),
        trials,
        np.random.default_rng(1),
        backend=NumpyBackend(),
        space=space,
        params={"limit": 10},
    )
    np.testing.assert_array_equal(new_pop, population)
    np.testing.assert_array_equal(new_trials, trials)


def test_employed_rejects_singleton_population() -> None:
    strategy = OriginalABC()
    with pytest.raises(SwarmSeekError, match="n >= 2"):
        strategy.employed_step(
            np.array([[0.0]]),
            np.array([0.0]),
            np.random.default_rng(0),
            backend=NumpyBackend(),
            space=ContinuousSpace([(0.0, 1.0)]),
            params={},
        )


def test_re_register_after_clear() -> None:
    clear_registry()
    with pytest.raises(UnknownVariantError):
        get_variant("original")
    register_variant("original", OriginalABC, replace=True)
    assert get_variant("original").name == "original"
