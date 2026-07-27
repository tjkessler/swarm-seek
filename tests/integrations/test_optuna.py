"""Tests for ABCSampler (requires Optuna)."""

from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

from swarm_seek.integrations import ABCSampler


def test_abcsampler_tiny_study() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -2.0, 2.0)
        y = trial.suggest_float("y", -2.0, 2.0)
        return x**2 + y**2

    sampler = ABCSampler(pop_size=4, limit=10, seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=20)
    assert study.best_value is not None
    assert study.best_value < 2.0
