"""Tests for PermutationSpace."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.errors import BoundsError
from swarm_seek.space import PermutationSpace


def test_sample_are_permutations() -> None:
    space = PermutationSpace(5)
    rng = np.random.default_rng(0)
    samples = space.sample(20, rng)
    assert samples.shape == (20, 5)
    for row in samples:
        space.validate(row)


def test_repair_restores_permutation() -> None:
    space = PermutationSpace(4)
    noisy = np.array([[0.1, 3.9, 1.2, 2.0], [10.0, -1.0, 0.0, 5.0]], dtype=np.float64)
    repaired = space.repair(noisy)
    assert repaired.shape == (2, 4)
    for row in repaired:
        space.validate(row)


def test_validate_rejects_duplicates() -> None:
    space = PermutationSpace(3)
    with pytest.raises(BoundsError, match="permutation"):
        space.validate(np.array([0.0, 0.0, 1.0], dtype=np.float64))


def test_n_too_small() -> None:
    with pytest.raises(BoundsError):
        PermutationSpace(1)


def test_sample_rejects_bad_n_and_rng() -> None:
    space = PermutationSpace(3)
    with pytest.raises(BoundsError):
        space.sample(0, np.random.default_rng(0))
    with pytest.raises(TypeError):
        space.sample(1, np.random.RandomState(0))  # type: ignore[arg-type]


def test_repair_validate_shape_errors() -> None:
    space = PermutationSpace(3)
    with pytest.raises(BoundsError):
        space.repair(np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(BoundsError):
        space.validate(np.zeros((2, 2)))
