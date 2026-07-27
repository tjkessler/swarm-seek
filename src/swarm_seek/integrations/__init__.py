"""Optional ecosystem adapters (scikit-learn, Optuna)."""

from __future__ import annotations

__all__ = ["ABCSampler", "ABCSearchCV"]


def __getattr__(name: str):
    if name == "ABCSearchCV":
        from swarm_seek.integrations.sklearn import ABCSearchCV

        return ABCSearchCV
    if name == "ABCSampler":
        from swarm_seek.integrations.optuna_sampler import ABCSampler

        return ABCSampler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
