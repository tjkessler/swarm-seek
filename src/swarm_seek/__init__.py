"""Swarm Seek: literature-validated Artificial Bee Colony optimization."""

from swarm_seek.abc import ABC
from swarm_seek.space import ContinuousSpace, PermutationSpace
from swarm_seek.types import Solution

__version__ = "0.2.0"

__all__ = [
    "ABC",
    "ContinuousSpace",
    "PermutationSpace",
    "Solution",
    "__version__",
]
