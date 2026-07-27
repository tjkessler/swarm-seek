"""Literature-oracle fixtures and runners for published ABC tables.

Oracles encode experimental settings and reported statistics from primary
sources. They do **not** claim bit-identical reproduction when papers omit RNG
seeds; assertions use documented tolerance bands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from swarm_seek import ABC, ContinuousSpace
from swarm_seek.benchmarks.functions import sphere
from swarm_seek.types import FloatArray


@dataclass(frozen=True, slots=True)
class LiteratureOracle:
    """One published table statistic with runnable experimental settings."""

    id: str
    citation: str
    doi: str
    locator: str
    variant: str
    function_name: str
    objective: Callable[[FloatArray], float | FloatArray]
    bounds: tuple[tuple[float, float], ...]
    pop_size: int
    limit: int
    max_evals: int
    n_runs: int
    statistic: str
    expected: float
    abs_tol: float
    notes: str
    variant_params: Mapping[str, float | int | str] = field(default_factory=dict)
    max_iters: int | None = None


# Karaboga & Akay (2009), Table 13, function No. 3 (Sphere).
# Experiments 1: population size 50, max evaluations 500_000, 30 runs;
# limit = SN * D (Eq. 9). Values below 1e-12 are reported as 0.
KARABOGA_AKAY_2009_T13_SPHERE = LiteratureOracle(
    id="karaboga_akay_2009_t13_sphere",
    citation=(
        "Karaboga, D., & Akay, B. (2009). A comparative study of artificial "
        "bee colony algorithm. Applied Mathematics and Computation, 214(1), "
        "108–132."
    ),
    doi="10.1016/j.amc.2009.03.090",
    locator="Table 13, function No. 3 (Sphere), ABC Mean column",
    variant="original",
    function_name="sphere",
    objective=sphere,
    bounds=tuple((-100.0, 100.0) for _ in range(30)),
    pop_size=50,
    limit=50 * 30,  # SN * D (Eq. 9); SN = population/food-source size
    max_evals=500_000,
    n_runs=30,
    statistic="mean_of_best",
    expected=0.0,
    abs_tol=1e-12,  # paper reports values < 1e-12 as 0
    notes=(
        "Paper omits RNG algorithm/seeds; runs use numpy Generator with "
        "distinct integer seeds. Common parameters from Experiments 1: "
        "population size 50 and maximum evaluation number 500000. "
        "CI may use fewer than 30 runs with the same per-run protocol "
        "(documented in the test)."
    ),
)


# Zhu & Kwong (2010), Table 3, Sphere (f3), D=30, GABC with C=1.5.
# §4.2: 400_000 FEs; population size 80; max generations 5000; 30 runs.
# Interpreting population size 80 as colony size 2*SN ⇒ SN = 40 food sources
# (5000 × 2 × 40 = 400_000). Paper omits abandonment limit; use SN*D.
ZHU_KWONG_2010_T3_SPHERE = LiteratureOracle(
    id="zhu_kwong_2010_t3_sphere",
    citation=(
        "Zhu, G., & Kwong, S. (2010). Gbest-guided artificial bee colony "
        "algorithm for numerical function optimization. Applied Mathematics "
        "and Computation, 217(7), 3166–3173."
    ),
    doi="10.1016/j.amc.2010.08.049",
    locator="Table 3, Sphere (f3), D=30, GABC (C=1.5) Mean column",
    variant="gabc",
    function_name="sphere",
    objective=sphere,
    bounds=tuple((-100.0, 100.0) for _ in range(30)),
    pop_size=40,  # SN; paper "population size" 80 = 2*SN
    limit=40 * 30,  # SN * D (paper omits limit; Karaboga convention)
    max_evals=400_000,
    max_iters=5000,
    n_runs=30,
    statistic="mean_of_best",
    expected=4.176106e-16,
    abs_tol=1e-14,  # CI upper bound; paper mean ≈ 4.18e-16
    variant_params={"C": 1.5},
    notes=(
        "Paper omits RNG seeds and abandonment limit. Food-source count SN=40 "
        "follows 400000 FEs with 5000 generations at ~2*SN evaluations/cycle "
        "(colony size 80). Limit set to SN*D=1200. Early-stop at error < 1e-20 "
        "is not reproduced; runs use max_evals/max_iters only. CI may use "
        "fewer than 30 runs with the same per-run protocol."
    ),
)


# Karaboga & Gorkemli (2014), Table 10, Sphere, qABC with r=1.
# §4: CS=50, max evaluations 500_000, 30 runs; limit = CS*D/2 (Eq. 9);
# values below 1e-15 reported as 0. Table 10 qABC column uses r=1
# (Rosenbrock/Schaffer means match Tables 2/4 r=1 rows; §4 text).
KARABOGA_GORKEMLI_2014_T10_SPHERE = LiteratureOracle(
    id="karaboga_gorkemli_2014_t10_sphere",
    citation=(
        "Karaboga, D., & Gorkemli, B. (2014). A quick artificial bee colony "
        "(qABC) algorithm and its performance on optimization problems. "
        "Applied Soft Computing, 23, 227–238."
    ),
    doi="10.1016/j.asoc.2014.06.035",
    locator="Table 10, Sphere, qABC Mean column (r=1)",
    variant="qabc",
    function_name="sphere",
    objective=sphere,
    bounds=tuple((-100.0, 100.0) for _ in range(30)),
    pop_size=25,  # SN; paper colony size CS=50 = 2*SN
    limit=(50 * 30) // 2,  # CS * D / 2 (Eq. 9)
    max_evals=500_000,
    n_runs=30,
    statistic="mean_of_best",
    expected=0.0,
    abs_tol=1e-15,  # paper reports values < 1e-15 as 0
    variant_params={"r": 1.0},
    notes=(
        "Paper omits RNG algorithm/seeds. Colony size CS=50 interpreted as "
        "2*SN food sources (SN=25); limit=CS*D/2=750. Table 10 state-of-art "
        "qABC column uses r=1. CI may use fewer than 30 runs with the same "
        "per-run protocol."
    ),
)


# Akay & Karaboga (2012), Table 2, Sphere, MR=0.5, SF=1, Limit=200.
# §5.1: population size 10, max evaluations 30_000, D=10, 30 runs.
# Colony size 10 interpreted as CS=2*SN ⇒ SN=5 (matches Table 8 FES
# accounting: Cycle×2×SN ≈ Max.FES). Search range ±100; paper init range
# [-100, 50] not reproduced (sample full search box).
AKAY_KARABOGA_2012_T2_SPHERE = LiteratureOracle(
    id="akay_karaboga_2012_t2_sphere",
    citation=(
        "Akay, B., & Karaboga, D. (2012). A modified artificial bee colony "
        "algorithm for real-parameter optimization. Information Sciences, "
        "192, 120–142."
    ),
    doi="10.1016/j.ins.2010.07.015",
    locator="Table 2, Sphere, ABC MR=0.5 / SF=1 / Limit=200 Mean column",
    variant="mabc",
    function_name="sphere",
    objective=sphere,
    bounds=tuple((-100.0, 100.0) for _ in range(10)),
    pop_size=5,  # SN; paper population/colony size 10 = 2*SN
    limit=200,
    max_evals=30_000,
    n_runs=30,
    statistic="mean_of_best",
    expected=9.63e-17,
    abs_tol=1e-14,  # CI upper bound; paper mean ≈ 9.63e-17
    variant_params={"MR": 0.5, "SF": 1.0},
    notes=(
        "Paper omits RNG seeds. Uses modified search (MR=0.5, SF=1) from the "
        "Table 2 MR sweep (SF:1, Limit=200). Population size 10 taken as "
        "colony size 2*SN (SN=5). Initialization range [-100, 50] from Table 1 "
        "is not reproduced; ContinuousSpace samples the search box ±100. "
        "CI may use fewer than 30 runs with the same per-run protocol."
    ),
)


# Reduced FE budgets for PR CI smokes (still assert fixture abs_tol on Sphere).
# Full paper max_evals remain on each LiteratureOracle and @pytest.mark.slow tests.
CI_SMOKE_MAX_EVALS: dict[str, int] = {
    "karaboga_akay_2009_t13_sphere": 150_000,
    "zhu_kwong_2010_t3_sphere": 100_000,
    "karaboga_gorkemli_2014_t10_sphere": 100_000,
    "akay_karaboga_2012_t2_sphere": 15_000,
}
CI_SMOKE_N_RUNS = 1


def mean_best_over_runs(
    oracle: LiteratureOracle,
    *,
    n_runs: int | None = None,
    seed0: int = 0,
    max_evals: int | None = None,
    max_iters: int | None = None,
) -> float:
    """Run ``oracle`` for ``n_runs`` seeds and return mean best fitness.

    Optional ``max_evals`` / ``max_iters`` override the fixture budgets for
    fast CI smokes; omit them to use the paper protocol stored on ``oracle``.
    """
    runs = oracle.n_runs if n_runs is None else int(n_runs)
    if runs < 1:
        msg = f"n_runs must be >= 1; got {runs}."
        raise ValueError(msg)

    evals = oracle.max_evals if max_evals is None else int(max_evals)
    iters = oracle.max_iters if max_iters is None else max_iters
    if evals < 1:
        msg = f"max_evals must be >= 1; got {evals}."
        raise ValueError(msg)

    space = ContinuousSpace(list(oracle.bounds))
    bests: list[float] = []
    for i in range(runs):
        colony = ABC(
            space,
            variant=oracle.variant,
            pop_size=oracle.pop_size,
            limit=oracle.limit,
            max_evals=evals,
            max_iters=iters,
            seed=seed0 + i,
            **dict(oracle.variant_params),
        )
        bests.append(float(colony.minimize(oracle.objective).fitness))
    return float(np.mean(bests))
