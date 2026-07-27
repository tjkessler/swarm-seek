"""Tests for ABCSearchCV (requires scikit-learn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from swarm_seek.integrations import ABCSearchCV


def test_abcsearchcv_fit_ridge() -> None:
    x, y = make_regression(n_samples=40, n_features=3, noise=0.1, random_state=0)
    search = ABCSearchCV(
        Ridge(),
        param_distributions={"alpha": [0.1, 1.0, 10.0]},
        n_iter=8,
        pop_size=4,
        limit=10,
        cv=3,
        seed=0,
    )
    search.fit(x, y)
    assert hasattr(search, "best_params_")
    assert np.isfinite(search.best_score_)
