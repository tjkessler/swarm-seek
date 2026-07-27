"""Optuna adapter: ``ABCSampler`` (optional ``swarm-seek[optuna]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from optuna.distributions import (
        BaseDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.samplers import BaseSampler
    from optuna.study import Study
    from optuna.trial import FrozenTrial, TrialState
except ImportError as exc:  # pragma: no cover
    _OPTUNA_IMPORT_ERROR = exc
    BaseSampler = object  # type: ignore[assignment,misc]
else:
    _OPTUNA_IMPORT_ERROR = None

from swarm_seek.abc import ABC
from swarm_seek.space import ContinuousSpace


def _require_optuna() -> None:
    if _OPTUNA_IMPORT_ERROR is not None:
        msg = (
            "ABCSampler requires the optional dependency 'optuna'. "
            "Install with: pip install 'swarm-seek[optuna]'."
        )
        raise ImportError(msg) from _OPTUNA_IMPORT_ERROR


def _bounds_from_search_space(
    search_space: dict[str, BaseDistribution],
) -> tuple[list[str], list[tuple[float, float]]]:
    keys: list[str] = []
    bounds: list[tuple[float, float]] = []
    for name, dist in search_space.items():
        if isinstance(dist, FloatDistribution):
            keys.append(name)
            bounds.append((float(dist.low), float(dist.high)))
        elif isinstance(dist, IntDistribution):
            keys.append(name)
            bounds.append((float(dist.low), float(dist.high) + 1.0 - 1e-9))
        else:
            msg = (
                f"ABCSampler supports FloatDistribution and IntDistribution; "
                f"got {type(dist).__name__} for {name!r}."
            )
            raise NotImplementedError(msg)
    return keys, bounds


def _decode(
    row: np.ndarray,
    keys: list[str],
    search_space: dict[str, BaseDistribution],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for i, name in enumerate(keys):
        dist = search_space[name]
        value = float(row[i])
        if isinstance(dist, IntDistribution):
            params[name] = int(np.clip(np.rint(value), dist.low, dist.high))
        else:
            params[name] = float(np.clip(value, dist.low, dist.high))
    return params


class ABCSampler(BaseSampler):
    """Optuna sampler driven by Swarm Seek ABC ask/tell.

    Relative search spaces of float/int distributions are mapped to a
    :class:`~swarm_seek.space.ContinuousSpace`. Each Optuna trial consumes one
    candidate from an ABC ``ask`` batch; ``after_trial`` feeds objective values
    back via ``tell``.
    """

    def __init__(
        self,
        *,
        pop_size: int = 10,
        limit: int = 20,
        variant: str = "original",
        seed: int | None = None,
    ) -> None:
        _require_optuna()
        self.pop_size = int(pop_size)
        self.limit = int(limit)
        self.variant = variant
        self.seed = seed
        self._opt: ABC | None = None
        self._keys: list[str] = []
        self._search_space: dict[str, BaseDistribution] = {}
        self._queue: list[dict[str, Any]] = []
        self._pending_scores: list[float] = []
        self._ask_size: int = 0
        self._sense: str = "minimize"
        self._rng = np.random.default_rng(seed)

    def reseed_rng(self) -> None:
        """Reseed the internal NumPy generator."""
        self._rng = np.random.default_rng(self.seed)

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        """Return the intersection search space of completed trials."""
        from optuna.search_space import intersection_search_space

        del trial
        return intersection_search_space(study.get_trials(deepcopy=False))

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Suggest a full parameter dict from the ABC colony."""
        del trial
        if not search_space:
            return {}
        self._sense = "maximize" if study.direction.name == "MAXIMIZE" else "minimize"
        self._maybe_tell()
        if self._opt is None or set(self._keys) != set(search_space):
            self._reset_colony(search_space)
        if not self._queue:
            self._refill_queue()
        if not self._queue:
            return self._uniform_sample(search_space)
        return self._queue.pop(0)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Fallback independent sampling (uniform in distribution bounds)."""
        del study, trial
        if isinstance(param_distribution, FloatDistribution):
            return float(
                self._rng.uniform(param_distribution.low, param_distribution.high)
            )
        if isinstance(param_distribution, IntDistribution):
            return int(
                self._rng.integers(param_distribution.low, param_distribution.high + 1)
            )
        msg = (
            f"ABCSampler independent sampling supports float/int distributions; "
            f"got {type(param_distribution).__name__} for {param_name!r}."
        )
        raise NotImplementedError(msg)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Record completed objective values for the next ``tell``."""
        del study, trial
        if state != TrialState.COMPLETE or values is None:
            return
        self._pending_scores.append(float(values[0]))

    def _reset_colony(self, search_space: dict[str, BaseDistribution]) -> None:
        keys, bounds = _bounds_from_search_space(search_space)
        space = ContinuousSpace(bounds)
        self._opt = ABC(
            space,
            variant=self.variant,
            pop_size=max(2, self.pop_size),
            limit=self.limit,
            sense=self._sense,  # type: ignore[arg-type]
            max_evals=10_000_000,
            seed=self.seed,
        )
        self._keys = keys
        self._search_space = dict(search_space)
        self._queue = []
        self._pending_scores = []
        self._ask_size = 0

    def _maybe_tell(self) -> None:
        if self._opt is None or self._ask_size == 0:
            return
        if len(self._pending_scores) < self._ask_size:
            return
        batch = np.asarray(self._pending_scores[: self._ask_size], dtype=np.float64)
        self._pending_scores = self._pending_scores[self._ask_size :]
        self._opt.tell(batch)
        self._ask_size = 0

    def _refill_queue(self) -> None:
        assert self._opt is not None
        if self._ask_size != 0:
            # Waiting on evaluations for the current ask batch.
            return
        if self._opt.converged:
            return
        batch = self._opt.ask()
        self._ask_size = int(batch.shape[0])
        if self._ask_size == 0:
            self._opt.tell([])
            self._ask_size = 0
            return
        for row in batch:
            self._queue.append(_decode(row, self._keys, self._search_space))

    def _uniform_sample(
        self,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, dist in search_space.items():
            params[name] = self.sample_independent(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                name,
                dist,
            )
        return params
