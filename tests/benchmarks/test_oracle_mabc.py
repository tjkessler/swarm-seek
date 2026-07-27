"""Literature oracle: MABC vs Akay & Karaboga (2012) Table 2."""

from __future__ import annotations

import pytest

from swarm_seek.benchmarks.oracles import (
    AKAY_KARABOGA_2012_T2_SPHERE,
    mean_best_over_runs,
)


def test_akay_karaboga_2012_table2_sphere_mean_best() -> None:
    """CI oracle: same protocol as Table 2 Sphere / MABC MR=0.5, fewer runs.

    The paper used 30 independent runs. This CI gate uses 5 runs with identical
    per-run settings and asserts the mean of best fitnesses is within the
    documented upper bound (``abs_tol``). Full ``n_runs=30`` is slow-marked.
    """
    oracle = AKAY_KARABOGA_2012_T2_SPHERE
    n_runs = 5
    mean_best = mean_best_over_runs(oracle, n_runs=n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed: mean best={mean_best!r} "
        f"(expected ~{oracle.expected} within abs_tol={oracle.abs_tol}). "
        f"Source: {oracle.citation} DOI {oracle.doi}; {oracle.locator}. "
        f"Settings: D={len(oracle.bounds)}, pop_size={oracle.pop_size}, "
        f"limit={oracle.limit}, max_evals={oracle.max_evals}, "
        f"MR={oracle.variant_params.get('MR')}, "
        f"SF={oracle.variant_params.get('SF')}, "
        f"n_runs={n_runs} (paper n_runs={oracle.n_runs}). "
        f"Notes: {oracle.notes}"
    )


@pytest.mark.slow
def test_akay_karaboga_2012_table2_sphere_mean_best_full_runs() -> None:
    """Full 30-run reproduction of Table 2 Sphere / MABC MR=0.5 mean-best."""
    oracle = AKAY_KARABOGA_2012_T2_SPHERE
    mean_best = mean_best_over_runs(oracle, n_runs=oracle.n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed (full runs): mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )
