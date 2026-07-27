"""Variant strategy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from swarm_seek.types import FloatArray

if TYPE_CHECKING:
    from swarm_seek.backends import BackendProtocol
    from swarm_seek.space import Space


@runtime_checkable
class VariantStrategy(Protocol):
    """Published update rules for employed / onlooker / scout phases."""

    name: str
    citation: str

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
        """Propose candidates from the employed-bee phase."""
        ...

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
        """Propose candidates from the onlooker-bee phase."""
        ...

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
        """Abandon exhausted sources and return updated population and trials."""
        ...
