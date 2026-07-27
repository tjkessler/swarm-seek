"""Literature oracle: GABC vs Zhu & Kwong (2010) Table 3."""

from __future__ import annotations

import pytest

from swarm_seek.benchmarks.oracles import (
    ZHU_KWONG_2010_T3_SPHERE,
    mean_best_over_runs,
)


def test_zhu_kwong_2010_table3_sphere_mean_best() -> None:
    """CI oracle: same protocol as Table 3 Sphere / GABC C=1.5, fewer runs.

    The paper used 30 independent runs. This CI gate uses 5 runs with identical
    per-run settings and asserts the mean of best fitnesses is within the
    documented upper bound (``abs_tol``). Full ``n_runs=30`` is slow-marked.
    """
    oracle = ZHU_KWONG_2010_T3_SPHERE
    n_runs = 5
    mean_best = mean_best_over_runs(oracle, n_runs=n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed: mean best={mean_best!r} "
        f"(expected ~{oracle.expected} within abs_tol={oracle.abs_tol}). "
        f"Source: {oracle.citation} DOI {oracle.doi}; {oracle.locator}. "
        f"Settings: D={len(oracle.bounds)}, pop_size={oracle.pop_size}, "
        f"limit={oracle.limit}, max_evals={oracle.max_evals}, "
        f"max_iters={oracle.max_iters}, C={oracle.variant_params.get('C')}, "
        f"n_runs={n_runs} (paper n_runs={oracle.n_runs}). "
        f"Notes: {oracle.notes}"
    )


@pytest.mark.slow
def test_zhu_kwong_2010_table3_sphere_mean_best_full_runs() -> None:
    """Full 30-run reproduction of Table 3 Sphere / GABC C=1.5 mean-best."""
    oracle = ZHU_KWONG_2010_T3_SPHERE
    mean_best = mean_best_over_runs(oracle, n_runs=oracle.n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed (full runs): mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )
