"""Parity and registry tests for Numba / JAX backends."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_seek.backends import NumpyBackend, get_backend
from swarm_seek.space import ContinuousSpace


def _fixture() -> tuple[
    ContinuousSpace, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    space = ContinuousSpace([(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)])
    population = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, -1.0, 0.5],
            [-2.0, 3.0, -0.5],
        ],
        dtype=np.float64,
    )
    partner_indices = np.array([1, 2, 0], dtype=np.int64)
    phi = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ref = NumpyBackend().generate_candidates(
        population, partner_indices, phi, space=space
    )
    return space, population, partner_indices, phi, ref


def test_numba_parity_with_numpy() -> None:
    pytest.importorskip("numba")
    space, pop, partners, phi, ref = _fixture()
    backend = get_backend("numba")
    assert backend.name == "numba"
    got = backend.generate_candidates(pop, partners, phi, space=space)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-12)


def test_jax_parity_with_numpy() -> None:
    pytest.importorskip("jax")
    space, pop, partners, phi, ref = _fixture()
    backend = get_backend("jax")
    assert backend.name == "jax"
    got = backend.generate_candidates(pop, partners, phi, space=space)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-10)


def test_auto_prefers_numba_when_installed() -> None:
    pytest.importorskip("numba")
    backend = get_backend("auto")
    assert backend.name == "numba"
