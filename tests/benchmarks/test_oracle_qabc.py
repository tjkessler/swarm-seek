"""Literature oracle: qABC vs Karaboga & Gorkemli (2014) Table 10."""

from __future__ import annotations

import pytest

from swarm_seek.benchmarks.oracles import (
    KARABOGA_GORKEMLI_2014_T10_SPHERE,
    mean_best_over_runs,
)


def test_karaboga_gorkemli_2014_table10_sphere_mean_best() -> None:
    """CI oracle: same protocol as Table 10 Sphere / qABC r=1, fewer runs.

    The paper used 30 independent runs. This CI gate uses 5 runs with identical
    per-run settings and asserts the mean of best fitnesses is within the
    paper's reporting threshold for ``0`` (``< 1e-15``). Full ``n_runs=30`` is
    covered by the slow test.
    """
    oracle = KARABOGA_GORKEMLI_2014_T10_SPHERE
    n_runs = 5
    mean_best = mean_best_over_runs(oracle, n_runs=n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed: mean best={mean_best!r} "
        f"(expected ~{oracle.expected} within abs_tol={oracle.abs_tol}). "
        f"Source: {oracle.citation} DOI {oracle.doi}; {oracle.locator}. "
        f"Settings: D={len(oracle.bounds)}, pop_size={oracle.pop_size}, "
        f"limit={oracle.limit}, max_evals={oracle.max_evals}, "
        f"r={oracle.variant_params.get('r')}, "
        f"n_runs={n_runs} (paper n_runs={oracle.n_runs}). "
        f"Notes: {oracle.notes}"
    )


@pytest.mark.slow
def test_karaboga_gorkemli_2014_table10_sphere_mean_best_full_runs() -> None:
    """Full 30-run reproduction of Table 10 Sphere / qABC r=1 mean-best."""
    oracle = KARABOGA_GORKEMLI_2014_T10_SPHERE
    mean_best = mean_best_over_runs(oracle, n_runs=oracle.n_runs, seed0=0)
    assert mean_best <= oracle.abs_tol, (
        f"Oracle {oracle.id} failed (full runs): mean best={mean_best!r}; "
        f"{oracle.locator}; DOI {oracle.doi}."
    )
