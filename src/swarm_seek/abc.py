"""Public Artificial Bee Colony façade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from swarm_seek.backends import BackendProtocol
from swarm_seek.colony import Colony
from swarm_seek.errors import SwarmSeekError
from swarm_seek.space import Space
from swarm_seek.types import FloatArray, Sense, Solution
from swarm_seek.variants import VariantStrategy


class ABC:
    """Ergonomic ask/tell façade around :class:`~swarm_seek.colony.Colony`.

    Parameters match the design sketch: choose a registered ``variant``, a
    ``space``, colony size / abandonment ``limit``, optimization ``sense``,
    convergence budgets, RNG ``seed``, and execution ``backend``. Extra keyword
    arguments are forwarded as variant parameters (e.g. ``C``, ``r``, ``MR``,
    ``SF``).

    Notes
    -----
    Power users may construct :class:`~swarm_seek.colony.Colony` directly.
    ``minimize`` is convenience sugar over the ask/tell loop.
    """

    def __init__(
        self,
        space: Space,
        *,
        variant: str = "original",
        strategy: VariantStrategy | None = None,
        backend: BackendProtocol | str = "auto",
        pop_size: int = 20,
        limit: int = 100,
        sense: Sense = "minimize",
        max_evals: int | None = None,
        max_iters: int | None = None,
        stall_evals: int | None = None,
        seed: int | None = None,
        **variant_params: Any,
    ) -> None:
        self._colony = Colony(
            space,
            variant=variant if strategy is None else None,
            strategy=strategy,
            backend=backend,
            pop_size=pop_size,
            limit=limit,
            sense=sense,
            max_evals=max_evals,
            max_iters=max_iters,
            stall_evals=stall_evals,
            seed=seed,
            **variant_params,
        )

    @property
    def colony(self) -> Colony:
        """Underlying :class:`~swarm_seek.colony.Colony` engine."""
        return self._colony

    @property
    def converged(self) -> bool:
        """Whether a configured stopping criterion has been met."""
        return self._colony.converged

    @property
    def best(self) -> Solution:
        """Best solution found so far."""
        return self._colony.best

    @property
    def n_evals(self) -> int:
        """Objective evaluations consumed."""
        return self._colony.n_evals

    @property
    def n_iters(self) -> int:
        """Completed employed→onlooker→scout cycles."""
        return self._colony.n_iters

    def ask(self) -> FloatArray:
        """Return candidates that need evaluation this step."""
        return self._colony.ask()

    def tell(self, fitnesses: FloatArray | list[float]) -> None:
        """Consume objective values for the last ``ask`` batch."""
        self._colony.tell(fitnesses)

    def minimize(
        self,
        objective: Callable[[FloatArray], FloatArray | list[float]],
        *,
        max_evals: int | None = None,
        max_iters: int | None = None,
        stall_evals: int | None = None,
    ) -> Solution:
        """Run ask/tell until convergence and return the best solution.

        Parameters
        ----------
        objective
            Callable mapping candidate array ``(n, d)`` to fitnesses ``(n,)``.
        max_evals, max_iters, stall_evals
            Optional budget overrides applied only when the colony has not yet
            started (no ``tell`` yet). If the run already started, existing
            colony budgets are used and overrides are rejected.

        Returns
        -------
        Solution
            Best solution at termination.
        """
        if max_evals is not None or max_iters is not None or stall_evals is not None:
            try:
                self._colony.configure_budgets(
                    max_evals=max_evals,
                    max_iters=max_iters,
                    stall_evals=stall_evals,
                )
            except SwarmSeekError as exc:
                msg = (
                    "Budget overrides to minimize() are only allowed before "
                    "the first ask/tell."
                )
                if "before the first ask" in str(exc):
                    raise SwarmSeekError(msg) from exc
                raise

        while not self._colony.converged:
            candidates = self._colony.ask()
            if candidates.size == 0:
                self._colony.tell([])
                continue
            fitnesses = np.asarray(objective(candidates), dtype=np.float64)
            self._colony.tell(fitnesses)
        return self._colony.best
