"""Smoke tests for package installability and public API exports."""

from __future__ import annotations

import importlib
import sys

import swarm_seek
from swarm_seek import ABC, ContinuousSpace, Solution


def test_version_is_pep440_prerelease() -> None:
    assert swarm_seek.__version__ == "0.1.0"


def test_planned_subpackages_importable() -> None:
    for name in (
        "swarm_seek.space",
        "swarm_seek.types",
        "swarm_seek.errors",
        "swarm_seek.backends",
        "swarm_seek.variants",
        "swarm_seek.colony",
        "swarm_seek.benchmarks",
        "swarm_seek.integrations",
    ):
        module = importlib.import_module(name)
        assert module.__doc__


def test_root_all_matches_design() -> None:
    assert swarm_seek.__all__ == [
        "ABC",
        "ContinuousSpace",
        "Solution",
        "__version__",
    ]
    assert ABC is swarm_seek.ABC
    assert ContinuousSpace is swarm_seek.ContinuousSpace
    assert Solution is swarm_seek.Solution


def test_import_does_not_require_optional_extras() -> None:
    optional = ("sklearn", "optuna", "numba", "jax")
    loaded = [name for name in optional if name in sys.modules]
    # Importing swarm_seek (already done above) must not have pulled these in.
    assert loaded == []
