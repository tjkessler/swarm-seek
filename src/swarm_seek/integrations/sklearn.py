"""scikit-learn adapter: ``ABCSearchCV`` (optional ``swarm-seek[sklearn]``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
except ImportError as exc:  # pragma: no cover
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

from swarm_seek.abc import ABC
from swarm_seek.space import ContinuousSpace


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        msg = (
            "ABCSearchCV requires the optional dependency 'scikit-learn'. "
            "Install with: pip install 'swarm-seek[sklearn]'."
        )
        raise ImportError(msg) from _SKLEARN_IMPORT_ERROR


def _distribution_bounds(
    param_distributions: Mapping[str, Any],
) -> tuple[list[str], list[tuple[float, float]], dict[str, Any]]:
    """Map RandomizedSearchCV-style distributions to a ContinuousSpace."""
    keys: list[str] = []
    bounds: list[tuple[float, float]] = []
    meta: dict[str, Any] = {}
    for key, dist in param_distributions.items():
        keys.append(key)
        if hasattr(dist, "rvs"):
            kwds = getattr(dist, "kwds", {}) or {}
            loc = float(kwds.get("loc", 0.0))
            scale = float(kwds.get("scale", 1.0))
            if scale <= 0.0:
                msg = f"scipy distribution for {key!r} must have positive scale."
                raise ValueError(msg)
            bounds.append((loc, loc + scale))
            meta[key] = {"kind": "scipy"}
        elif isinstance(dist, (list, tuple, np.ndarray)):
            values = list(dist)
            if not values:
                msg = f"param_distributions[{key!r}] must be non-empty."
                raise ValueError(msg)
            bounds.append((0.0, float(len(values)) - 1e-9))
            meta[key] = {"kind": "categorical", "values": values}
        else:
            msg = (
                f"Unsupported distribution for {key!r}: {type(dist)!r}. "
                "Use a scipy.stats distribution or a non-empty list."
            )
            raise TypeError(msg)
    return keys, bounds, meta


def _decode_vector(
    row: np.ndarray,
    keys: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for i, key in enumerate(keys):
        info = meta[key]
        value = float(row[i])
        if info["kind"] == "categorical":
            idx = int(np.clip(np.floor(value), 0, len(info["values"]) - 1))
            params[key] = info["values"][idx]
        else:
            params[key] = value
    return params


class ABCSearchCV:
    """Hyperparameter search via Swarm Seek ABC ask/tell.

    RandomizedSearchCV-like surface: ``fit`` runs cross-validated scoring
    owned by scikit-learn; ABC only proposes parameter vectors.

    Parameters
    ----------
    estimator
        scikit-learn estimator.
    param_distributions
        Dict of parameter names mapped to scipy.stats distributions or lists
        of categorical values.
    n_iter
        Evaluation budget forwarded as ``max_evals`` to ABC.
    pop_size, limit, variant, seed
        Forwarded to :class:`~swarm_seek.abc.ABC`.
    scoring, n_jobs, refit, cv
        Passed to :func:`~sklearn.model_selection.cross_val_score`.
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Mapping[str, Any],
        *,
        n_iter: int = 10,
        pop_size: int = 10,
        limit: int = 20,
        variant: str = "original",
        seed: int | None = None,
        scoring: Any = None,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: Any = None,
    ) -> None:
        _require_sklearn()
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.pop_size = int(pop_size)
        self.limit = int(limit)
        self.variant = variant
        self.seed = seed
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = bool(refit)
        self.cv = cv

    def fit(self, X: Any, y: Any = None) -> ABCSearchCV:
        """Run ABC-guided CV search and optionally refit the best estimator."""
        keys, bounds, meta = _distribution_bounds(self.param_distributions)
        if not keys:
            msg = "param_distributions must contain at least one parameter."
            raise ValueError(msg)
        space = ContinuousSpace(bounds)
        pop_size = min(max(2, self.pop_size), max(2, self.n_iter))
        opt = ABC(
            space,
            variant=self.variant,
            pop_size=pop_size,
            limit=self.limit,
            sense="maximize",
            max_evals=self.n_iter,
            seed=self.seed,
        )

        params_list: list[dict[str, Any]] = []
        scores: list[float] = []

        while not opt.converged:
            batch = opt.ask()
            if batch.shape[0] == 0:
                opt.tell([])
                continue
            batch_scores: list[float] = []
            for row in batch:
                params = _decode_vector(row, keys, meta)
                est = clone(self.estimator).set_params(**params)
                cv_scores = cross_val_score(
                    est,
                    X,
                    y,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                )
                mean_score = float(np.mean(cv_scores))
                if not np.isfinite(mean_score):
                    mean_score = -np.inf
                params_list.append(params)
                scores.append(mean_score)
                batch_scores.append(mean_score)
            opt.tell(np.asarray(batch_scores, dtype=np.float64))

        score_arr = np.asarray(scores, dtype=np.float64)
        best_idx = int(np.argmax(score_arr))
        self.cv_results_ = {
            "params": params_list,
            "mean_test_score": score_arr,
        }
        self.best_index_ = best_idx
        self.best_params_ = params_list[best_idx]
        self.best_score_ = float(score_arr[best_idx])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y)
        return self
