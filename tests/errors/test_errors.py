"""Tests for typed Swarm Seek exceptions."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import swarm_seek.errors as errors_pkg
import swarm_seek.types as types_pkg
from swarm_seek.errors import (
    BoundsError,
    SolutionValidationError,
    SwarmSeekError,
    TellShapeError,
    UnknownBackendError,
    UnknownVariantError,
)


def test_exception_hierarchy() -> None:
    assert issubclass(BoundsError, SwarmSeekError)
    assert issubclass(TellShapeError, SwarmSeekError)
    assert issubclass(UnknownVariantError, SwarmSeekError)
    assert issubclass(UnknownBackendError, SwarmSeekError)
    assert issubclass(SolutionValidationError, SwarmSeekError)
    assert issubclass(BoundsError, ValueError)
    assert issubclass(UnknownVariantError, KeyError)
    assert issubclass(UnknownBackendError, KeyError)


def test_exceptions_are_raiseable_with_message() -> None:
    with pytest.raises(BoundsError, match="bounds"):
        raise BoundsError("invalid bounds")


def test_power_user_exports() -> None:
    assert errors_pkg.BoundsError is BoundsError
    assert "Solution" in types_pkg.__all__


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_types_and_errors_do_not_import_higher_layers() -> None:
    """L0 modules must not import backends, variants, or colony."""
    root = Path(__file__).resolve().parents[2] / "src" / "swarm_seek"
    forbidden = ("swarm_seek.backends", "swarm_seek.variants", "swarm_seek.colony")
    for package in ("types", "errors"):
        for path in (root / package).rglob("*.py"):
            imported = _imported_modules(path)
            for name in forbidden:
                assert name not in imported, f"{path} imports {name}"
                assert not any(mod.startswith(f"{name}.") for mod in imported)
