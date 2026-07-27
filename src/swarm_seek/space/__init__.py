"""Search-space representations (continuous and combinatorial)."""

from swarm_seek.space.continuous import ContinuousSpace
from swarm_seek.space.permutation import PermutationSpace
from swarm_seek.space.protocol import Space

__all__ = [
    "ContinuousSpace",
    "PermutationSpace",
    "Space",
]
