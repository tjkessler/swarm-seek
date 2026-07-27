"""Tests for :class:`swarm_seek.types.Solution`."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.errors import SolutionValidationError
from swarm_seek.types import Solution


def test_solution_accepts_1d_finite_vector() -> None:
    sol = Solution(x=np.array([1.0, 2.0], dtype=np.float64), fitness=3.5)
    assert sol.x.shape == (2,)
    assert sol.x.dtype == np.float64
    assert sol.fitness == 3.5


def test_solution_coerces_list_input() -> None:
    sol = Solution(x=[0.0, -1.5], fitness=0)
    assert isinstance(sol.x, np.ndarray)
    assert sol.x.tolist() == [0.0, -1.5]
    assert sol.fitness == 0.0


def test_solution_is_frozen() -> None:
    sol = Solution(x=np.zeros(2), fitness=0.0)
    with pytest.raises(AttributeError):
        sol.fitness = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("x", "fitness"),
    [
        (np.zeros((2, 2)), 0.0),
        (np.array([]), 0.0),
        (np.array([np.nan]), 0.0),
        (np.array([np.inf]), 0.0),
        (np.zeros(2), np.nan),
        (np.zeros(2), np.inf),
        (np.zeros(2), "bad"),
    ],
)
def test_solution_rejects_invalid_inputs(x: object, fitness: object) -> None:
    with pytest.raises(SolutionValidationError):
        Solution(x=x, fitness=fitness)  # type: ignore[arg-type]
