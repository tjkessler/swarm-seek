"""Typed exceptions for the public and power-user APIs."""

from swarm_seek.errors.exceptions import (
    BoundsError,
    SolutionValidationError,
    SwarmSeekError,
    TellShapeError,
    UnknownBackendError,
    UnknownVariantError,
)

__all__ = [
    "BoundsError",
    "SolutionValidationError",
    "SwarmSeekError",
    "TellShapeError",
    "UnknownBackendError",
    "UnknownVariantError",
]
