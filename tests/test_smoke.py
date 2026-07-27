"""Smoke tests for package installability and public API exports."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import swarm_seek
from swarm_seek import ABC, ContinuousSpace, PermutationSpace, Solution

_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_pep440() -> None:
    assert swarm_seek.__version__ == "0.2.0"


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
        "PermutationSpace",
        "Solution",
        "__version__",
    ]
    assert ABC is swarm_seek.ABC
    assert ContinuousSpace is swarm_seek.ContinuousSpace
    assert PermutationSpace is swarm_seek.PermutationSpace
    assert Solution is swarm_seek.Solution


def test_import_does_not_require_optional_extras() -> None:
    """Fresh interpreter import of swarm_seek must not load optional extras."""
    code = (
        "import sys; import swarm_seek; "
        "opt=('sklearn','optuna','numba','jax'); "
        "loaded=[n for n in opt if n in sys.modules]; "
        "assert loaded==[], loaded"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()})},
    )
    assert proc.returncode == 0, proc.stderr
