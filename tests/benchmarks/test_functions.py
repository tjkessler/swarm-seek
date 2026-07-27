"""Tests for classical benchmark objectives."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.benchmarks import ackley, griewank, rastrigin, rosenbrock, sphere
from swarm_seek.errors import SwarmSeekError


@pytest.mark.parametrize(
    "func",
    [sphere, rastrigin, griewank, ackley],
)
def test_origin_is_global_minimum(func: object) -> None:
    x = np.zeros(5)
    assert func(x) == pytest.approx(0.0, abs=1e-12)  # type: ignore[operator]


def test_rosenbrock_ones_is_global_minimum() -> None:
    assert rosenbrock(np.ones(4)) == pytest.approx(0.0, abs=1e-12)


def test_rosenbrock_rejects_1d() -> None:
    with pytest.raises(SwarmSeekError, match="d >= 2"):
        rosenbrock(np.array([1.0]))


def test_vectorized_batch_matches_rowwise() -> None:
    rng = np.random.default_rng(0)
    batch = rng.uniform(-2.0, 2.0, size=(7, 3))
    for func in (sphere, rastrigin, griewank, ackley, rosenbrock):
        got = np.asarray(func(batch), dtype=np.float64)
        expected = np.array([float(func(row)) for row in batch])
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_sphere_known_value() -> None:
    assert sphere(np.array([3.0, 4.0])) == pytest.approx(25.0)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(SwarmSeekError):
        sphere(np.zeros((2, 2, 2)))
    with pytest.raises(SwarmSeekError):
        sphere(np.array([]))
