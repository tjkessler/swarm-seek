"""Tests for Gbest-guided ABC update rules."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend
from swarm_seek.errors import SwarmSeekError
from swarm_seek.space import ContinuousSpace
from swarm_seek.variants import get_variant, register_variant
from swarm_seek.variants.gabc import GbestABC, _gabc_batch, _require_c


@pytest.fixture(autouse=True)
def _ensure_gabc_registered() -> None:
    register_variant("gabc", GbestABC, replace=True)
    yield


def test_get_variant_gabc() -> None:
    strategy = get_variant("gabc")
    assert isinstance(strategy, GbestABC)
    assert strategy.name == "gabc"
    assert "Zhu" in strategy.citation


def test_gabc_hand_checked_equation(monkeypatch: pytest.MonkeyPatch) -> None:
    space = ContinuousSpace([(-20.0, 20.0), (-20.0, 20.0)])
    population = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, -1.0]], dtype=np.float64)
    # Source 0 is global best under minimize.
    fitness = np.array([0.0, 10.0, 5.0])

    import swarm_seek.variants.gabc as gabc_mod

    monkeypatch.setattr(
        gabc_mod,
        "partner_indices",
        lambda n_pop, _rng: np.array([1, 2, 0], dtype=np.int64),
    )

    def fixed_weights(n_pop, n_dim, _rng, *, low, high, dims=None):
        if dims is None:
            chosen = np.array([0, 1, 0], dtype=np.int64)
            weights = np.zeros((n_pop, n_dim), dtype=np.float64)
            if low < 0:
                # phi
                weights[np.arange(n_pop), chosen] = np.array([0.5, -0.5, 1.0])
            else:
                weights[np.arange(n_pop), chosen] = np.array([0.0, 0.0, 0.0])
            return weights, chosen
        chosen = np.asarray(dims, dtype=np.int64)
        weights = np.zeros((n_pop, n_dim), dtype=np.float64)
        # psi on same dims
        weights[np.arange(n_pop), chosen] = np.array([0.0, 1.0, 0.5])
        return weights, chosen

    monkeypatch.setattr(gabc_mod, "sparse_weights", fixed_weights)

    got = _gabc_batch(
        population,
        fitness,
        np.random.default_rng(0),
        space=space,
        sense="minimize",
        c_val=1.5,
    )

    # gbest = [0, 0]
    # v0: dim0: 0 + 0.5*(0-2) + 0*(0-0) = -1
    # v1: dim1: 4 + (-0.5)*(4-(-1)) + 1.0*(0-4) = 4 - 2.5 - 4 = -2.5
    # v2: dim0: 1 + 1.0*(1-0) + 0.5*(0-1) = 1 + 1 - 0.5 = 1.5
    expected = np.array([[-1.0, 0.0], [2.0, -2.5], [1.5, -1.0]], dtype=np.float64)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_gbest_term_pulls_toward_best(monkeypatch: pytest.MonkeyPatch) -> None:
    space = ContinuousSpace([(-50.0, 50.0)])
    # Two sources on a line; best at 0, other at 10.
    population = np.array([[0.0], [10.0]], dtype=np.float64)
    fitness = np.array([0.0, 100.0])

    import swarm_seek.variants.gabc as gabc_mod

    monkeypatch.setattr(
        gabc_mod,
        "partner_indices",
        lambda n_pop, _rng: np.array([1, 0], dtype=np.int64),
    )

    def weights_phi_only(n_pop, n_dim, _rng, *, low, high, dims=None):
        chosen = np.zeros(n_pop, dtype=np.int64) if dims is None else np.asarray(dims)
        w = np.zeros((n_pop, n_dim), dtype=np.float64)
        if low < 0:
            w[:, 0] = 0.0  # no neighbor displacement
        else:
            w[:, 0] = 1.0  # psi = 1 on the mutated dim
        return w, chosen.astype(np.int64)

    monkeypatch.setattr(gabc_mod, "sparse_weights", weights_phi_only)

    out = _gabc_batch(
        population,
        fitness,
        np.random.default_rng(0),
        space=space,
        sense="minimize",
        c_val=1.5,
    )
    # Row 1: 10 + 0*(10-0) + 1*(0-10) = 0
    np.testing.assert_allclose(out[1], [0.0], atol=1e-12)


def test_employed_deterministic_under_seed() -> None:
    space = ContinuousSpace([(-5.0, 5.0)] * 2)
    strategy = GbestABC()
    population = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    fitness = np.array([1.0, 3.0, 0.2])
    kwargs = dict(backend=NumpyBackend(), space=space, params={"C": 1.5})
    a = strategy.employed_step(population, fitness, np.random.default_rng(11), **kwargs)
    b = strategy.employed_step(population, fitness, np.random.default_rng(11), **kwargs)
    np.testing.assert_allclose(a, b)


def test_scout_abandons() -> None:
    strategy = GbestABC()
    space = ContinuousSpace([(0.0, 1.0)])
    population = np.array([[0.1], [0.9]], dtype=np.float64)
    trials = np.array([0, 3], dtype=np.int64)
    new_pop, new_trials = strategy.scout_step(
        population,
        np.zeros(2),
        trials,
        np.random.default_rng(0),
        backend=NumpyBackend(),
        space=space,
        params={"limit": 3},
    )
    assert new_trials[1] == 0
    assert new_trials[0] == 0
    assert 0.0 <= float(new_pop[1, 0]) <= 1.0


def test_require_c_validation() -> None:
    assert _require_c({}) == 1.5
    assert _require_c({"C": 2}) == 2.0
    with pytest.raises(SwarmSeekError, match="C"):
        _require_c({"C": 0})
    with pytest.raises(SwarmSeekError, match="C"):
        _require_c({"C": "nope"})


def test_onlooker_runs_and_stays_in_bounds() -> None:
    strategy = GbestABC()
    space = ContinuousSpace([(-1.0, 1.0), (-1.0, 1.0)])
    population = np.array(
        [[0.0, 0.0], [0.5, -0.5], [-0.25, 0.75]],
        dtype=np.float64,
    )
    fitness = np.array([0.1, 2.0, 1.0])
    out = strategy.onlooker_step(
        population,
        fitness,
        np.random.default_rng(3),
        backend=NumpyBackend(),
        space=space,
        params={"sense": "minimize", "C": 1.5},
    )
    assert out.shape == population.shape
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


def test_employed_rejects_bad_fitness_shape() -> None:
    strategy = GbestABC()
    with pytest.raises(SwarmSeekError, match="fitness must have shape"):
        strategy.employed_step(
            np.array([[0.0], [1.0]]),
            np.zeros(3),
            np.random.default_rng(0),
            backend=NumpyBackend(),
            space=ContinuousSpace([(0.0, 1.0)]),
            params={},
        )
