"""Swarm Seek: literature-validated Artificial Bee Colony optimization."""

from swarm_seek.abc import ABC
from swarm_seek.space import ContinuousSpace
from swarm_seek.types import Solution

__version__ = "0.1.0a0"

__all__ = [
    "ABC",
    "ContinuousSpace",
    "Solution",
    "__version__",
]
