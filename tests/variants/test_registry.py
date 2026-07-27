"""Tests for variant protocol helpers and registry."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_seek.backends import BackendProtocol
from swarm_seek.errors import SwarmSeekError, UnknownVariantError
from swarm_seek.space import Space
from swarm_seek.types import FloatArray
from swarm_seek.variants import (
    VARIANT_NAMES,
    VariantStrategy,
    clear_registry,
    default_params,
    get_variant,
    list_variants,
    register_variant,
    require_limit,
)


class _FakeStrategy:
    name = "original"
    citation = "fake-test"

    def employed_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> FloatArray:
        return np.asarray(population, dtype=np.float64).copy()

    def onlooker_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> FloatArray:
        return np.asarray(population, dtype=np.float64).copy()

    def scout_step(
        self,
        population: FloatArray,
        fitness: FloatArray,
        trials: NDArray[np.integer],
        rng: np.random.Generator,
        *,
        backend: BackendProtocol,
        space: Space,
        params: dict[str, float | int | str],
    ) -> tuple[FloatArray, NDArray[np.integer]]:
        return (
            np.asarray(population, dtype=np.float64).copy(),
            np.asarray(trials).copy(),
        )


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()
    # Restore builtins so later test modules still see shipped variants.
    from swarm_seek.variants.gabc import GbestABC
    from swarm_seek.variants.mabc import ModifiedABC
    from swarm_seek.variants.original import OriginalABC
    from swarm_seek.variants.qabc import QuickABC

    register_variant("original", OriginalABC, replace=True)
    register_variant("gabc", GbestABC, replace=True)
    register_variant("qabc", QuickABC, replace=True)
    register_variant("mabc", ModifiedABC, replace=True)


def test_variant_names_match_design() -> None:
    assert VARIANT_NAMES == ("original", "gabc", "qabc", "mabc")


def test_register_and_get_variant() -> None:
    register_variant("original", _FakeStrategy)
    strategy = get_variant("original")
    assert isinstance(strategy, VariantStrategy)
    assert strategy.name == "original"
    assert list_variants() == ("original",)


def test_unknown_variant_raises() -> None:
    with pytest.raises(UnknownVariantError, match="Unknown variant"):
        get_variant("original")


def test_duplicate_registration_raises_unless_replace() -> None:
    register_variant("gabc", _FakeStrategy)
    with pytest.raises(SwarmSeekError, match="already registered"):
        register_variant("gabc", _FakeStrategy)
    register_variant("gabc", _FakeStrategy, replace=True)
    assert get_variant("gabc").citation == "fake-test"


def test_require_limit_helpers() -> None:
    assert require_limit({}) == 100
    assert require_limit(default_params(limit=25)) == 25
    with pytest.raises(SwarmSeekError, match=">= 1"):
        require_limit({"limit": 0})
    with pytest.raises(SwarmSeekError, match="integer"):
        require_limit({"limit": "nope"})  # type: ignore[dict-item]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_variants_do_not_import_colony_or_integrations() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "swarm_seek" / "variants"
    forbidden = ("swarm_seek.colony", "swarm_seek.integrations")
    for path in root.rglob("*.py"):
        imported = _imported_modules(path)
        for name in forbidden:
            assert name not in imported
            assert not any(mod.startswith(f"{name}.") for mod in imported)
