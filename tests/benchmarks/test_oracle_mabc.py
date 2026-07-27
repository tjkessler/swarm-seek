"""Literature oracle: MABC vs Akay & Karaboga (2012) Table 2."""

from __future__ import annotations

import pytest

from swarm_seek.benchmarks.oracles import (
    AKAY_KARABOGA_2012_T2_SPHERE,
    CI_SMOKE_MAX_EVALS,
    CI_SMOKE_N_RUNS,
    mean_best_over_runs,
)


def test_akay_karaboga_2012_table2_sphere_mean_best() -> None:
    """CI smoke: paper abs_tol with reduced max_evals (not full FE budget)."""
    oracle = AKAY_KARABOGA_2012_T2_SPHERE
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
def test_akay_karaboga_2012_table2_sphere_mean_best_paper_protocol() -> None:
    """Paper FE budget, 5 runs (PR-default formerly; now slow)."""
    oracle = AKAY_KARABOGA_2012_T2_SPHERE
    n_runs = 5
    mean_best = mean_best_over_runs(oracle, n_runs=n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} paper-protocol failed: mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )


@pytest.mark.slow
def test_akay_karaboga_2012_table2_sphere_mean_best_full_runs() -> None:
    """Full 30-run reproduction of Table 2 Sphere / MABC mean-best."""
    oracle = AKAY_KARABOGA_2012_T2_SPHERE
    mean_best = mean_best_over_runs(oracle, n_runs=oracle.n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed (full runs): mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )
