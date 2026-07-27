"""Tests for :class:`swarm_seek.space.ContinuousSpace`."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.errors import BoundsError
from swarm_seek.space import ContinuousSpace, Space


def test_continuous_space_from_sequence_of_pairs() -> None:
    space = ContinuousSpace([(-1.0, 1.0), (0.0, 5.0)])
    assert space.n_dim == 2
    assert isinstance(space, Space)
    np.testing.assert_allclose(space.lower, [-1.0, 0.0])
    np.testing.assert_allclose(space.upper, [1.0, 5.0])


def test_continuous_space_from_ndarray() -> None:
    bounds = np.array([[-2.0, 2.0], [-3.0, 3.0], [0.0, 1.0]])
    space = ContinuousSpace(bounds)
    assert space.n_dim == 3


@pytest.mark.parametrize(
    "bounds",
    [
        [],
        [(-1.0,)],
        [(-1.0, -1.0)],
        [(-1.0, 1.0), (np.nan, 1.0)],
        [(-1.0, 1.0), (2.0, 1.0)],
        np.zeros(3),
        np.zeros((2, 3)),
    ],
)
def test_invalid_bounds_raise(bounds: object) -> None:
    with pytest.raises(BoundsError):
        ContinuousSpace(bounds)  # type: ignore[arg-type]


def test_sample_shape_bounds_and_seed_determinism() -> None:
    space = ContinuousSpace([(-1.0, 1.0), (0.0, 2.0)])
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    a = space.sample(5, rng_a)
    b = space.sample(5, rng_b)
    assert a.shape == (5, 2)
    np.testing.assert_allclose(a, b)
    assert np.all(a >= space.lower)
    assert np.all(a <= space.upper)


def test_sample_rejects_non_positive_n_and_bad_rng() -> None:
    space = ContinuousSpace([(-1.0, 1.0)])
    rng = np.random.default_rng(1)
    with pytest.raises(BoundsError, match="n must be"):
        space.sample(0, rng)
    with pytest.raises(TypeError, match="Generator"):
        space.sample(1, np.random.RandomState(0))  # type: ignore[arg-type]


def test_repair_clips_out_of_box() -> None:
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    raw = np.array([[-1.0, 0.5], [0.2, 3.0], [0.1, 0.9]])
    repaired = space.repair(raw)
    np.testing.assert_allclose(repaired, [[0.0, 0.5], [0.2, 1.0], [0.1, 0.9]])
    vector = space.repair(np.array([-2.0, 4.0]))
    np.testing.assert_allclose(vector, [0.0, 1.0])


def test_validate_accepts_in_box_and_rejects_oob() -> None:
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    space.validate(np.array([0.0, 1.0]))
    space.validate(np.array([[0.25, 0.75], [1.0, 0.0]]))
    with pytest.raises(BoundsError, match="outside"):
        space.validate(np.array([-0.01, 0.5]))
    with pytest.raises(BoundsError, match="shape"):
        space.validate(np.array([0.0, 0.0, 0.0]))


def test_empty_bounds_matrix_and_empty_point_batch_raise() -> None:
    with pytest.raises(BoundsError, match="at least one dimension"):
        ContinuousSpace(np.empty((0, 2)))
    space = ContinuousSpace([(0.0, 1.0)])
    with pytest.raises(BoundsError, match="at least one point"):
        space.validate(np.empty((0, 1)))
    with pytest.raises(BoundsError, match="expected shape"):
        space.repair(np.zeros((2, 2, 2)))
