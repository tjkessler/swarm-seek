"""Tests for the public ABC façade."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek import ABC
from swarm_seek.colony import Colony
from swarm_seek.errors import SwarmSeekError
from swarm_seek.space import ContinuousSpace
from swarm_seek.types import Solution
from swarm_seek.variants import VARIANT_NAMES


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1)


CONTINUOUS_VARIANTS = [v for v in VARIANT_NAMES if v != "cabc"]


@pytest.mark.parametrize("variant", CONTINUOUS_VARIANTS)
def test_abc_constructs_all_variants(variant: str) -> None:
    space = ContinuousSpace([(-2.0, 2.0)] * 2)
    extra: dict[str, float] = {}
    if variant == "gabc":
        extra["C"] = 1.5
    elif variant == "qabc":
        extra["r"] = 1.0
    elif variant == "mabc":
        extra["MR"] = 0.4
        extra["SF"] = 1.0
    abc = ABC(
        space,
        variant=variant,
        pop_size=8,
        max_evals=40,
        seed=0,
        **extra,
    )
    assert isinstance(abc.colony, Colony)
    result = abc.minimize(_sphere)
    assert isinstance(result, Solution)
    assert result.x.shape == (2,)


def test_minimize_sphere_original_and_gabc() -> None:
    space = ContinuousSpace([(-5.0, 5.0)] * 2)
    for variant, extra in (
        ("original", {}),
        ("gabc", {"C": 1.5}),
    ):
        abc = ABC(
            space,
            variant=variant,
            pop_size=12,
            limit=20,
            max_evals=2_000,
            seed=1,
            **extra,
        )
        best = abc.minimize(_sphere)
        assert best.fitness < 1e-2


def test_ask_tell_delegates() -> None:
    abc = ABC(
        ContinuousSpace([(-1.0, 1.0)] * 2),
        pop_size=4,
        max_evals=30,
        seed=0,
    )
    while not abc.converged:
        x = abc.ask()
        abc.tell(_sphere(x) if x.size else [])
    assert abc.n_evals >= 30
    assert abc.best.fitness >= 0.0


def test_minimize_budget_override_before_start() -> None:
    abc = ABC(
        ContinuousSpace([(-1.0, 1.0)] * 2),
        pop_size=4,
        max_evals=10_000,
        seed=0,
    )
    best = abc.minimize(_sphere, max_evals=20)
    assert isinstance(best, Solution)
    assert abc.n_evals >= 20


def test_minimize_budget_override_after_start_raises() -> None:
    abc = ABC(
        ContinuousSpace([(-1.0, 1.0)] * 2),
        pop_size=4,
        max_evals=100,
        seed=0,
    )
    abc.tell(_sphere(abc.ask()))
    with pytest.raises(SwarmSeekError, match="Budget overrides"):
        abc.minimize(_sphere, max_evals=50)
