"""Typed exceptions for Swarm Seek."""

from __future__ import annotations


class SwarmSeekError(Exception):
    """Base exception for all Swarm Seek errors."""


class SolutionValidationError(SwarmSeekError, ValueError):
    """Raised when a :class:`~swarm_seek.types.Solution` is invalid."""


class BoundsError(SwarmSeekError, ValueError):
    """Raised when search-space bounds are invalid or violated."""


class TellShapeError(SwarmSeekError, ValueError):
    """Raised when ``tell`` fitnesses do not match the last ``ask`` batch."""


class UnknownVariantError(SwarmSeekError, KeyError):
    """Raised when a variant name is not registered."""


class UnknownBackendError(SwarmSeekError, KeyError):
    """Raised when a backend name is not registered."""
