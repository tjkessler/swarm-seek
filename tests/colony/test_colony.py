"""Tests for the Colony ask/tell engine."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend
from swarm_seek.colony import Colony
from swarm_seek.errors import SwarmSeekError, TellShapeError
from swarm_seek.space import ContinuousSpace
from swarm_seek.variants import OriginalABC


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1)


def _make_colony(**kwargs: object) -> Colony:
    space = ContinuousSpace([(-5.0, 5.0), (-5.0, 5.0)])
    defaults: dict[str, object] = {
        "space": space,
        "variant": "original",
        "backend": "numpy",
        "pop_size": 6,
        "limit": 5,
        "max_evals": 200,
        "seed": 0,
    }
    defaults.update(kwargs)
    return Colony(**defaults)  # type: ignore[arg-type]


def test_ask_tell_round_trip_original() -> None:
    colony = _make_colony(max_evals=80, seed=1)
    while not colony.converged:
        x = colony.ask()
        colony.tell(_sphere(x) if x.size else [])
    assert colony.best.fitness < 10.0
    assert colony.n_evals >= 80 or colony.n_iters > 0


def test_tell_shape_mismatch_raises() -> None:
    colony = _make_colony()
    x = colony.ask()
    with pytest.raises(TellShapeError, match="tell expected"):
        colony.tell(np.zeros(x.shape[0] + 1))


def test_seeded_runs_are_deterministic() -> None:
    def run(seed: int) -> tuple[float, list[np.ndarray]]:
        colony = _make_colony(seed=seed, max_iters=2, max_evals=None)
        asks: list[np.ndarray] = []
        while not colony.converged:
            x = colony.ask()
            asks.append(np.asarray(x).copy())
            colony.tell(_sphere(x) if x.size else [])
        return float(colony.best.fitness), asks

    fit_a, asks_a = run(7)
    fit_b, asks_b = run(7)
    assert fit_a == fit_b
    assert len(asks_a) == len(asks_b)
    for a, b in zip(asks_a, asks_b, strict=True):
        np.testing.assert_allclose(a, b)


def test_sense_maximize_prefers_higher_fitness() -> None:
    colony = _make_colony(sense="maximize", max_iters=3, max_evals=None, seed=2)

    def objective(x: np.ndarray) -> np.ndarray:
        # Higher near origin for maximize sense with negated sphere.
        return -_sphere(x)

    while not colony.converged:
        x = colony.ask()
        colony.tell(objective(x) if x.size else [])
    # Best fitness should be the largest (least negative) among evaluated.
    assert colony.best.fitness <= 0.0


def test_max_evals_convergence() -> None:
    colony = _make_colony(max_evals=12, max_iters=None, pop_size=4, seed=0)
    while not colony.converged:
        x = colony.ask()
        colony.tell(_sphere(x) if x.size else [])
    assert colony.n_evals >= 12
    assert colony.converged


def test_max_iters_convergence() -> None:
    colony = _make_colony(max_iters=2, max_evals=None, pop_size=4, seed=0)
    while not colony.converged:
        x = colony.ask()
        colony.tell(_sphere(x) if x.size else [])
    assert colony.n_iters >= 2
    assert colony.converged


def test_stall_evals_convergence() -> None:
    # Constant objective → no improvement after init → stall trips.
    colony = _make_colony(
        max_evals=10_000,
        stall_evals=8,
        pop_size=4,
        seed=0,
    )
    while not colony.converged:
        x = colony.ask()
        colony.tell(np.ones(x.shape[0]) if x.size else [])
    assert colony.converged
    assert colony.n_evals >= 8


def test_empty_scout_batch_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    colony = _make_colony(max_iters=1, max_evals=None, limit=1000, pop_size=4)

    # Force scout to never abandon.
    def no_abandon(population, fitness, trials, rng, *, backend, space, params):
        pop = np.asarray(population, dtype=np.float64).copy()
        return pop, np.asarray(trials).copy()

    monkeypatch.setattr(colony._strategy, "scout_step", no_abandon)

    # init
    colony.tell(_sphere(colony.ask()))
    # employed
    colony.tell(_sphere(colony.ask()))
    # onlooker
    x = colony.ask()
    colony.tell(_sphere(x) if x.size else [])
    # scout — empty
    assert colony.phase == "scout"
    empty = colony.ask()
    assert empty.shape == (0, 2)
    colony.tell([])
    assert colony.phase == "employed"
    assert colony.n_iters == 1


def test_double_ask_without_tell_raises() -> None:
    colony = _make_colony()
    colony.ask()
    with pytest.raises(SwarmSeekError, match="tell"):
        colony.ask()


def test_requires_budget() -> None:
    with pytest.raises(SwarmSeekError, match="max_evals or max_iters"):
        Colony(
            ContinuousSpace([(-1.0, 1.0)] * 2),
            max_evals=None,
            max_iters=None,
        )


def test_strategy_kwarg() -> None:
    colony = Colony(
        ContinuousSpace([(-1.0, 1.0)] * 2),
        strategy=OriginalABC(),
        backend=NumpyBackend(),
        pop_size=4,
        max_evals=20,
        seed=0,
    )
    colony.tell(_sphere(colony.ask()))
    assert colony.best.x.shape == (2,)


def test_best_before_tell_raises() -> None:
    colony = _make_colony()
    with pytest.raises(SwarmSeekError, match="initial tell"):
        _ = colony.best


def test_pop_size_must_be_at_least_two() -> None:
    with pytest.raises(SwarmSeekError, match="pop_size"):
        _make_colony(pop_size=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_evals": 0, "max_iters": None}, "max_evals"),
        ({"max_evals": None, "max_iters": 0}, "max_iters"),
        ({"max_evals": 10, "stall_evals": 0}, "stall_evals"),
        ({"max_evals": 10, "sense": "sideways"}, "sense"),
    ],
)
def test_invalid_constructor_kwargs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(SwarmSeekError, match=match):
        _make_colony(**kwargs)


def test_requires_variant_or_strategy() -> None:
    with pytest.raises(SwarmSeekError, match="variant"):
        Colony(
            ContinuousSpace([(-1.0, 1.0)] * 2),
            variant=None,
            strategy=None,
            max_evals=10,
        )


def test_tell_without_ask_raises() -> None:
    colony = _make_colony()
    with pytest.raises(SwarmSeekError, match="ask"):
        colony.tell([1.0])


def test_tell_rejects_non_finite_fitness() -> None:
    colony = _make_colony()
    x = colony.ask()
    bad = np.full(x.shape[0], np.nan)
    with pytest.raises(TellShapeError, match="finite"):
        colony.tell(bad)


def test_configure_budgets_before_start() -> None:
    colony = _make_colony(max_evals=50, max_iters=None, stall_evals=None)
    colony.configure_budgets(max_evals=20, max_iters=3, stall_evals=5)
    while not colony.converged:
        x = colony.ask()
        colony.tell(_sphere(x) if x.size else [])
    assert colony.converged
    assert colony.n_evals >= 20 or colony.n_iters >= 3


def test_configure_budgets_after_start_raises() -> None:
    colony = _make_colony()
    colony.tell(_sphere(colony.ask()))
    with pytest.raises(SwarmSeekError, match="configure_budgets"):
        colony.configure_budgets(max_evals=10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_evals": 0}, "max_evals"),
        ({"max_iters": 0}, "max_iters"),
        ({"stall_evals": 0}, "stall_evals"),
    ],
)
def test_configure_budgets_rejects_non_positive(
    kwargs: dict[str, object], match: str
) -> None:
    colony = _make_colony(max_evals=10)
    with pytest.raises(SwarmSeekError, match=match):
        colony.configure_budgets(**kwargs)  # type: ignore[arg-type]


def test_configure_budgets_cannot_leave_no_stop_criterion() -> None:
    # Public kwargs only set budgets (never clear). Exercise the defensive
    # empty-budget guard by clearing both fields before configure_budgets.
    colony = Colony(
        ContinuousSpace([(-1.0, 1.0)] * 2),
        variant="original",
        pop_size=4,
        max_evals=10,
        max_iters=None,
        seed=0,
    )
    colony._max_evals = None
    colony._max_iters = None
    with pytest.raises(SwarmSeekError, match="max_evals or max_iters"):
        colony.configure_budgets(stall_evals=5)
