"""Tests for the NumPy backend and registry."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from swarm_seek.backends import BackendProtocol, NumpyBackend, get_backend
from swarm_seek.errors import BoundsError
from swarm_seek.space import ContinuousSpace


def _slow_generate(
    population: np.ndarray,
    partner_indices: np.ndarray,
    phi: np.ndarray,
    space: ContinuousSpace,
) -> np.ndarray:
    out = np.empty_like(population, dtype=np.float64)
    for i in range(population.shape[0]):
        partner = population[int(partner_indices[i])]
        row = np.empty(population.shape[1], dtype=np.float64)
        for j in range(population.shape[1]):
            row[j] = population[i, j] + phi[i, j] * (population[i, j] - partner[j])
        out[i] = space.repair(row)
    return out


def test_numpy_backend_matches_slow_reference() -> None:
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
    backend = NumpyBackend()
    assert isinstance(backend, BackendProtocol)
    got = backend.generate_candidates(population, partner_indices, phi, space=space)
    expected = _slow_generate(population, partner_indices, phi, space)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    # Original population must remain unchanged.
    np.testing.assert_array_equal(
        population,
        np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, -1.0, 0.5],
                [-2.0, 3.0, -0.5],
            ]
        ),
    )


def test_generate_candidates_repairs_out_of_bounds() -> None:
    space = ContinuousSpace([(0.0, 1.0), (0.0, 1.0)])
    population = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    partner_indices = np.array([1, 0], dtype=np.int64)
    # Large phi drives coordinates outside the box.
    phi = np.array([[10.0, 0.0], [0.0, -10.0]], dtype=np.float64)
    out = NumpyBackend().generate_candidates(
        population, partner_indices, phi, space=space
    )
    assert out.shape == (2, 2)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_shape_mismatches_raise() -> None:
    space = ContinuousSpace([(0.0, 1.0)])
    backend = NumpyBackend()
    with pytest.raises(BoundsError, match="2-D"):
        backend.generate_candidates(
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
            np.zeros((2, 1)),
            space=space,
        )
    population = np.array([[0.1], [0.2]], dtype=np.float64)
    with pytest.raises(BoundsError):
        backend.generate_candidates(
            population,
            np.array([0], dtype=np.int64),
            np.zeros_like(population),
            space=space,
        )
    with pytest.raises(BoundsError):
        backend.generate_candidates(
            population,
            np.array([0, 1], dtype=np.int64),
            np.zeros((2, 2)),
            space=space,
        )
    with pytest.raises(BoundsError):
        backend.generate_candidates(
            np.zeros((0, 1)),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 1)),
            space=space,
        )


def test_get_backend_numpy() -> None:
    a = get_backend("numpy")
    assert isinstance(a, NumpyBackend)
    assert a.name == "numpy"


def test_get_backend_auto_resolves() -> None:
    b = get_backend("auto")
    assert b.name in {"numpy", "numba"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_backends_do_not_import_variants_or_colony() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "swarm_seek" / "backends"
    forbidden = ("swarm_seek.variants", "swarm_seek.colony")
    for path in root.rglob("*.py"):
        imported = _imported_modules(path)
        for name in forbidden:
            assert name not in imported
            assert not any(mod.startswith(f"{name}.") for mod in imported)
