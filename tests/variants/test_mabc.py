"""Tests for Modified ABC update rules."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend
from swarm_seek.errors import SwarmSeekError
from swarm_seek.space import ContinuousSpace
from swarm_seek.variants import get_variant, register_variant
from swarm_seek.variants.mabc import (
    ModifiedABC,
    _mabc_batch,
    _require_mr,
    _require_sf,
    modification_mask,
)


@pytest.fixture(autouse=True)
def _ensure_mabc_registered() -> None:
    register_variant("mabc", ModifiedABC, replace=True)
    yield


def test_get_variant_mabc() -> None:
    strategy = get_variant("mabc")
    assert isinstance(strategy, ModifiedABC)
    assert strategy.name == "mabc"
    assert "Akay" in strategy.citation


def test_modification_mask_mr_one_mutates_all() -> None:
    mask = modification_mask(5, 4, np.random.default_rng(0), mr=1.0)
    assert mask.shape == (5, 4)
    assert np.all(mask)


def test_modification_mask_mr_zero_forces_one_dim() -> None:
    mask = modification_mask(8, 3, np.random.default_rng(1), mr=0.0)
    assert np.all(np.count_nonzero(mask, axis=1) == 1)


def test_sf_scales_phi_magnitude(monkeypatch: pytest.MonkeyPatch) -> None:
    space = ContinuousSpace([(-50.0, 50.0), (-50.0, 50.0)])
    population = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float64)

    import swarm_seek.variants.mabc as mabc_mod

    monkeypatch.setattr(
        mabc_mod,
        "partner_indices",
        lambda n_pop, _rng: np.array([1, 0], dtype=np.int64),
    )
    monkeypatch.setattr(
        mabc_mod,
        "modification_mask",
        lambda n_pop, n_dim, _rng, *, mr: np.ones((n_pop, n_dim), dtype=bool),
    )

    class _Rng:
        def uniform(self, low, high, size=None):
            # Always return +SF magnitude at the upper end.
            return np.full(size, high, dtype=np.float64)

        def integers(self, *args, **kwargs):  # pragma: no cover - unused
            raise AssertionError("integers should not be called")

        def random(self, *args, **kwargs):  # pragma: no cover - unused
            raise AssertionError("random should not be called")

    out = _mabc_batch(
        population,
        _Rng(),  # type: ignore[arg-type]
        backend=NumpyBackend(),
        space=space,
        mr=1.0,
        sf=2.0,
    )
    # v0 = [0,0] + 2*([0,0]-[2,4]) = [-4, -8]
    # v1 = [2,4] + 2*([2,4]-[0,0]) = [6, 12]
    np.testing.assert_allclose(out, [[-4.0, -8.0], [6.0, 12.0]], atol=1e-12)


def test_employed_deterministic_under_seed() -> None:
    strategy = ModifiedABC()
    space = ContinuousSpace([(-5.0, 5.0)] * 3)
    population = np.array(
        [[0.0, 1.0, -1.0], [2.0, 0.0, 0.5], [-1.0, -2.0, 1.5]],
        dtype=np.float64,
    )
    kwargs = dict(
        backend=NumpyBackend(),
        space=space,
        params={"MR": 0.4, "SF": 1.0},
    )
    a = strategy.employed_step(
        population, np.zeros(3), np.random.default_rng(21), **kwargs
    )
    b = strategy.employed_step(
        population, np.zeros(3), np.random.default_rng(21), **kwargs
    )
    np.testing.assert_allclose(a, b)


def test_onlooker_stays_in_bounds() -> None:
    strategy = ModifiedABC()
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    population = np.array([[0.1, 0.2], [0.8, 0.9], [0.4, 0.5]], dtype=np.float64)
    out = strategy.onlooker_step(
        population,
        np.array([0.0, 1.0, 0.5]),
        np.random.default_rng(4),
        backend=NumpyBackend(),
        space=space,
        params={"MR": 0.8, "SF": 1.5, "sense": "minimize"},
    )
    assert out.shape == population.shape
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_scout_abandons() -> None:
    strategy = ModifiedABC()
    space = ContinuousSpace([(0.0, 1.0)])
    population = np.array([[0.2], [0.7]], dtype=np.float64)
    trials = np.array([0, 2], dtype=np.int64)
    _, new_trials = strategy.scout_step(
        population,
        np.zeros(2),
        trials,
        np.random.default_rng(0),
        backend=NumpyBackend(),
        space=space,
        params={"limit": 2},
    )
    assert new_trials[1] == 0
    assert new_trials[0] == 0


def test_param_validation() -> None:
    assert _require_mr({}) == 0.4
    assert _require_sf({}) == 1.0
    with pytest.raises(SwarmSeekError, match="MR"):
        _require_mr({"MR": 1.5})
    with pytest.raises(SwarmSeekError, match="SF"):
        _require_sf({"SF": 0})
