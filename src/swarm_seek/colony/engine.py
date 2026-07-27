"""Ask/tell colony engine for Artificial Bee Colony variants.

Phase accounting
----------------
The colony advances through a fixed phase cycle. Each ``ask`` / ``tell`` pair
completes one phase:

1. **INIT** — first ``ask`` samples the population; ``tell`` stores initial
   fitness values. No employed/onlooker/scout operators run yet.
2. **EMPLOYED** — ``ask`` proposes one neighbor per food source via
   ``strategy.employed_step``; ``tell`` applies sense-aware greedy selection
   and trial updates.
3. **ONLOOKER** — ``ask`` proposes onlooker candidates and returns only rows
   that differ from the current population (Q5); ``tell`` applies greedy
   selection on that subset.
4. **SCOUT** — ``ask`` runs ``strategy.scout_step`` and returns only
   resampled (abandoned) sources; ``tell`` installs their fitness. A completed
   scout phase increments ``n_iters`` and returns to **EMPLOYED**.

An empty ``ask`` batch (shape ``(0, n_dim)``) is valid for onlooker/scout when
nothing needs evaluation; ``tell`` of an empty array advances the phase.

Convergence (Q4)
----------------
``converged`` is true when any configured criterion fires: ``max_evals``,
``max_iters``, and/or ``stall_evals`` (evaluations since the last improvement
of ``best``). At least one of ``max_evals`` or ``max_iters`` is required at
construction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from swarm_seek.backends import BackendProtocol, get_backend
from swarm_seek.errors import SwarmSeekError, TellShapeError
from swarm_seek.space import Space
from swarm_seek.types import FloatArray, Sense, Solution
from swarm_seek.variants import VariantStrategy, get_variant


class _Phase(Enum):
    INIT = "init"
    EMPLOYED = "employed"
    ONLOOKER = "onlooker"
    SCOUT = "scout"


class Colony:
    """Population state and ask/tell orchestration for one ABC run.

    Parameters
    ----------
    space
        Search space providing ``sample`` / ``repair``.
    variant
        Registered variant name (ignored if ``strategy`` is given).
    strategy
        Explicit strategy instance; takes precedence over ``variant``.
    backend
        Backend instance or name (``\"numpy\"`` / ``\"auto\"``).
    pop_size
        Number of food sources (must be ``>= 2``).
    limit
        Scout abandonment limit.
    sense
        ``\"minimize\"`` or ``\"maximize\"``.
    max_evals
        Optional evaluation budget.
    max_iters
        Optional iteration budget (employed→onlooker→scout cycles).
    stall_evals
        Optional stall window in evaluations without best improvement.
    seed
        Seed for ``numpy.random.Generator``.
    **variant_params
        Extra strategy parameters (e.g. ``C``, ``r``, ``MR``, ``SF``).
    """

    def __init__(
        self,
        space: Space,
        *,
        variant: str | None = "original",
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
        if pop_size < 2:
            msg = f"pop_size must be >= 2; got {pop_size}."
            raise SwarmSeekError(msg)
        if max_evals is None and max_iters is None:
            msg = "At least one of max_evals or max_iters must be set."
            raise SwarmSeekError(msg)
        if max_evals is not None and max_evals < 1:
            msg = f"max_evals must be >= 1 when set; got {max_evals}."
            raise SwarmSeekError(msg)
        if max_iters is not None and max_iters < 1:
            msg = f"max_iters must be >= 1 when set; got {max_iters}."
            raise SwarmSeekError(msg)
        if stall_evals is not None and stall_evals < 1:
            msg = f"stall_evals must be >= 1 when set; got {stall_evals}."
            raise SwarmSeekError(msg)
        if sense not in {"minimize", "maximize"}:
            msg = f"sense must be 'minimize' or 'maximize'; got {sense!r}."
            raise SwarmSeekError(msg)
        if strategy is None:
            if variant is None:
                msg = "Provide variant=... or strategy=...."
                raise SwarmSeekError(msg)
            strategy = get_variant(variant)
        if isinstance(backend, str):
            backend = get_backend(backend)

        self._space = space
        self._strategy = strategy
        self._backend = backend
        self._pop_size = int(pop_size)
        self._sense: Sense = sense
        self._max_evals = max_evals
        self._max_iters = max_iters
        self._stall_evals = stall_evals
        self._rng = np.random.default_rng(seed)
        self._params: dict[str, float | int | str] = {
            "limit": int(limit),
            "sense": sense,
            **variant_params,
        }

        self._phase = _Phase.INIT
        self._population: FloatArray | None = None
        self._fitness: FloatArray | None = None
        self._trials: NDArray[np.int64] | None = None
        self._n_evals = 0
        self._n_iters = 0
        self._evals_since_improve = 0
        self._best: Solution | None = None

        self._pending_idx: NDArray[np.int64] | None = None
        self._pending_x: FloatArray | None = None
        self._scout_population: FloatArray | None = None
        self._scout_trials: NDArray[np.int64] | None = None

    @property
    def n_evals(self) -> int:
        """Number of objective evaluations consumed via ``tell``."""
        return self._n_evals

    @property
    def n_iters(self) -> int:
        """Completed employed→onlooker→scout cycles."""
        return self._n_iters

    @property
    def best(self) -> Solution:
        """Best solution found so far.

        Raises
        ------
        SwarmSeekError
            If no fitness has been told yet.
        """
        if self._best is None:
            msg = "No solution available until the initial tell() completes."
            raise SwarmSeekError(msg)
        return self._best

    @property
    def converged(self) -> bool:
        """Return whether any configured stopping criterion has been met."""
        if self._max_evals is not None and self._n_evals >= self._max_evals:
            return True
        if self._max_iters is not None and self._n_iters >= self._max_iters:
            return True
        return (
            self._stall_evals is not None
            and self._best is not None
            and self._evals_since_improve >= self._stall_evals
        )

    @property
    def phase(self) -> str:
        """Current phase name (power-user introspection)."""
        return self._phase.value

    def configure_budgets(
        self,
        *,
        max_evals: int | None = None,
        max_iters: int | None = None,
        stall_evals: int | None = None,
    ) -> None:
        """Update convergence budgets before the first ``ask`` / ``tell``.

        Raises
        ------
        SwarmSeekError
            If the colony has already started, or if both ``max_evals`` and
            ``max_iters`` would be unset.
        """
        if self._phase is not _Phase.INIT or self._n_evals > 0:
            msg = "configure_budgets() is only allowed before the first ask()."
            raise SwarmSeekError(msg)
        if max_evals is not None:
            if max_evals < 1:
                msg = f"max_evals must be >= 1 when set; got {max_evals}."
                raise SwarmSeekError(msg)
            self._max_evals = int(max_evals)
        if max_iters is not None:
            if max_iters < 1:
                msg = f"max_iters must be >= 1 when set; got {max_iters}."
                raise SwarmSeekError(msg)
            self._max_iters = int(max_iters)
        if stall_evals is not None:
            if stall_evals < 1:
                msg = f"stall_evals must be >= 1 when set; got {stall_evals}."
                raise SwarmSeekError(msg)
            self._stall_evals = int(stall_evals)
        if self._max_evals is None and self._max_iters is None:
            msg = "At least one of max_evals or max_iters must be set."
            raise SwarmSeekError(msg)

    def ask(self) -> FloatArray:
        """Return candidates that need objective evaluation this step.

        Returns
        -------
        FloatArray
            Array of shape ``(n_ask, n_dim)`` (may be empty).

        Raises
        ------
        SwarmSeekError
            If a previous ``ask`` is still awaiting ``tell``.
        """
        if self._pending_idx is not None:
            msg = "tell() must be called before the next ask()."
            raise SwarmSeekError(msg)

        if self._phase is _Phase.INIT:
            self._population = self._space.sample(self._pop_size, self._rng)
            idx = np.arange(self._pop_size, dtype=np.int64)
            self._set_pending(idx, self._population.copy())
            return self._pending_x  # type: ignore[return-value]

        assert self._population is not None
        assert self._fitness is not None
        assert self._trials is not None

        if self._phase is _Phase.EMPLOYED:
            candidates = self._strategy.employed_step(
                self._population,
                self._fitness,
                self._rng,
                backend=self._backend,
                space=self._space,
                params=self._params,
            )
            idx = np.arange(self._pop_size, dtype=np.int64)
            self._set_pending(idx, np.asarray(candidates, dtype=np.float64))
            return self._pending_x  # type: ignore[return-value]

        if self._phase is _Phase.ONLOOKER:
            candidates = np.asarray(
                self._strategy.onlooker_step(
                    self._population,
                    self._fitness,
                    self._rng,
                    backend=self._backend,
                    space=self._space,
                    params=self._params,
                ),
                dtype=np.float64,
            )
            changed = ~np.all(
                np.isclose(candidates, self._population, rtol=0.0, atol=0.0),
                axis=1,
            )
            idx = np.nonzero(changed)[0].astype(np.int64)
            self._set_pending(idx, candidates[idx] if idx.size else candidates[:0])
            return self._pending_x  # type: ignore[return-value]

        # SCOUT
        new_pop, new_trials = self._strategy.scout_step(
            self._population,
            self._fitness,
            self._trials,
            self._rng,
            backend=self._backend,
            space=self._space,
            params=self._params,
        )
        new_pop = np.asarray(new_pop, dtype=np.float64)
        new_trials = np.asarray(new_trials, dtype=np.int64)
        changed = ~np.all(
            np.isclose(new_pop, self._population, rtol=0.0, atol=0.0),
            axis=1,
        )
        idx = np.nonzero(changed)[0].astype(np.int64)
        self._scout_population = new_pop
        self._scout_trials = new_trials
        self._set_pending(idx, new_pop[idx] if idx.size else new_pop[:0])
        return self._pending_x  # type: ignore[return-value]

    def tell(self, fitnesses: FloatArray | list[float]) -> None:
        """Consume objective values for the last ``ask`` batch.

        Parameters
        ----------
        fitnesses
            Vector of length ``n_ask`` matching the last ``ask`` order.

        Raises
        ------
        TellShapeError
            If length/shape does not match the pending ask batch.
        SwarmSeekError
            If ``tell`` is called without a pending ``ask``.
        """
        if self._pending_idx is None or self._pending_x is None:
            msg = "ask() must be called before tell()."
            raise SwarmSeekError(msg)

        fit = np.asarray(fitnesses, dtype=np.float64)
        n_ask = int(self._pending_idx.size)
        if fit.ndim != 1 or fit.shape[0] != n_ask:
            msg = f"tell expected fitness shape ({n_ask},); got {fit.shape}."
            raise TellShapeError(msg)
        if n_ask and not np.all(np.isfinite(fit)):
            msg = "tell fitnesses must be finite."
            raise TellShapeError(msg)

        prev_best = None if self._best is None else float(self._best.fitness)

        if self._phase is _Phase.INIT:
            self._fitness = np.empty(self._pop_size, dtype=np.float64)
            self._fitness[self._pending_idx] = fit
            self._trials = np.zeros(self._pop_size, dtype=np.int64)
            self._update_best()
            self._phase = _Phase.EMPLOYED
        elif self._phase is _Phase.EMPLOYED:
            self._greedy(self._pending_idx, self._pending_x, fit)
            self._phase = _Phase.ONLOOKER
        elif self._phase is _Phase.ONLOOKER:
            self._greedy(self._pending_idx, self._pending_x, fit)
            self._phase = _Phase.SCOUT
        else:  # SCOUT
            assert self._scout_population is not None
            assert self._scout_trials is not None
            assert self._fitness is not None
            self._population = self._scout_population
            self._trials = self._scout_trials
            if n_ask:
                self._fitness[self._pending_idx] = fit
            self._scout_population = None
            self._scout_trials = None
            self._update_best()
            self._phase = _Phase.EMPLOYED
            self._n_iters += 1

        self._n_evals += n_ask
        self._note_improvement(prev_best, n_ask)
        self._clear_pending()

    def _set_pending(self, idx: NDArray[np.int64], candidates: FloatArray) -> None:
        self._pending_idx = np.asarray(idx, dtype=np.int64)
        if self._pending_idx.size == 0:
            n_dim = self._space.n_dim
            self._pending_x = np.zeros((0, n_dim), dtype=np.float64)
        else:
            self._pending_x = np.asarray(candidates, dtype=np.float64)

    def _clear_pending(self) -> None:
        self._pending_idx = None
        self._pending_x = None

    def _is_better(self, new: float, old: float) -> bool:
        if self._sense == "minimize":
            return new < old
        return new > old

    def _greedy(
        self,
        indices: NDArray[np.int64],
        candidates: FloatArray,
        fitnesses: FloatArray,
    ) -> None:
        assert self._population is not None
        assert self._fitness is not None
        assert self._trials is not None
        for row, idx in enumerate(indices):
            i = int(idx)
            new_f = float(fitnesses[row])
            if self._is_better(new_f, float(self._fitness[i])):
                self._population[i] = candidates[row]
                self._fitness[i] = new_f
                self._trials[i] = 0
            else:
                self._trials[i] += 1
        self._update_best()

    def _update_best(self) -> None:
        assert self._population is not None
        assert self._fitness is not None
        if self._sense == "minimize":
            i = int(np.argmin(self._fitness))
        else:
            i = int(np.argmax(self._fitness))
        self._best = Solution(
            x=self._population[i].copy(),
            fitness=float(self._fitness[i]),
        )

    def _note_improvement(self, prev_best: float | None, n_ask: int) -> None:
        if self._best is None:
            return
        if prev_best is None or self._is_better(float(self._best.fitness), prev_best):
            self._evals_since_improve = 0
        else:
            self._evals_since_improve += n_ask
