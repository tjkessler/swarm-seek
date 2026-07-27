"""Literature oracle: Original ABC vs Karaboga & Akay (2009) Table 13."""

from __future__ import annotations

import pytest

from swarm_seek.benchmarks.oracles import (
    CI_SMOKE_MAX_EVALS,
    CI_SMOKE_N_RUNS,
    KARABOGA_AKAY_2009_T13_SPHERE,
    mean_best_over_runs,
)


def test_karaboga_akay_2009_table13_sphere_mean_best() -> None:
    """CI smoke: paper abs_tol with reduced max_evals (not full FE budget)."""
    oracle = KARABOGA_AKAY_2009_T13_SPHERE
    max_evals = CI_SMOKE_MAX_EVALS[oracle.id]
    mean_best = mean_best_over_runs(
        oracle,
        n_runs=CI_SMOKE_N_RUNS,
        seed0=0,
        max_evals=max_evals,
        max_iters=None,
    )
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} CI smoke failed: mean best={mean_best!r} "
        f"(expected ~{oracle.expected} within abs_tol={oracle.abs_tol}). "
        f"Source: {oracle.citation} DOI {oracle.doi}; {oracle.locator}. "
        f"CI settings: max_evals={max_evals} (paper {oracle.max_evals}), "
        f"n_runs={CI_SMOKE_N_RUNS} (paper {oracle.n_runs}). "
        f"Notes: {oracle.notes}"
    )


@pytest.mark.slow
def test_karaboga_akay_2009_table13_sphere_mean_best_paper_protocol() -> None:
    """Paper FE budget, 5 runs (PR-default formerly; now slow)."""
    oracle = KARABOGA_AKAY_2009_T13_SPHERE
    n_runs = 5
    mean_best = mean_best_over_runs(oracle, n_runs=n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} paper-protocol failed: mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )


@pytest.mark.slow
def test_karaboga_akay_2009_table13_sphere_mean_best_full_runs() -> None:
    """Full 30-run reproduction of Table 13 Sphere / ABC mean-best."""
    oracle = KARABOGA_AKAY_2009_T13_SPHERE
    mean_best = mean_best_over_runs(oracle, n_runs=oracle.n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed (full runs): mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )
