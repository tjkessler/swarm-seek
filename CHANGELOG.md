# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial package skeleton (`src/swarm_seek/` layout, Apache-2.0 license).
- Development tooling: pytest + coverage gate (90%), ruff, pre-commit, GitHub Actions CI.
- Sphinx + Furo documentation scaffold and Read the Docs configuration.
- Contribution and governance files (`CONTRIBUTING.md`, code of conduct, security policy, citation metadata).
- L0 types and errors: frozen `Solution`, shared aliases (`FloatArray`, `VariantName`, `BackendName`, `Sense`), and typed exceptions (`SwarmSeekError` hierarchy).
- `ContinuousSpace` with box bounds, uniform `sample`, clip `repair`, and `validate` (`Space` protocol).
- NumPy execution backend (`generate_candidates` + `get_backend("numpy"|"auto")`) and `UnknownBackendError`.
- Variant strategy protocol, shared params helpers, and registry (`register_variant` / `get_variant`).
- Original ABC strategy (`variant="original"`) with citation-linked employed / onlooker / scout steps.
- Gbest-guided ABC strategy (`variant="gabc"`, Zhu & Kwong 2010) with parameter `C`.
- Quick ABC strategy (`variant="qabc"`, Karaboga & Gorkemli 2014) with neighborhood radius `r`.
- Modified ABC strategy (`variant="mabc"`, Akay & Karaboga 2012) with parameters `MR` and `SF`.
- `Colony` ask/tell engine with phase accounting, sense-aware greedy selection, and `max_evals` / `max_iters` / `stall_evals` convergence.
- Public `ABC` façade with ask/tell delegation and `minimize(objective)` sugar for all four variants.
- Root exports: `ABC`, `ContinuousSpace`, `Solution`, `__version__`.
- Benchmark objectives: Sphere, Rastrigin, Rosenbrock, Griewank, Ackley (vectorized).
- Literature oracle for Original ABC vs Karaboga & Akay (2009) Table 13 Sphere (mean best).
- Literature oracle for GABC vs Zhu & Kwong (2010) Table 3 Sphere, ``C=1.5`` (mean best).
- Literature oracle for qABC vs Karaboga & Gorkemli (2014) Table 10 Sphere, ``r=1`` (mean best).
- Literature oracle for MABC vs Akay & Karaboga (2012) Table 2 Sphere, ``MR=0.5``, ``SF=1`` (mean best).
- Example notebooks under ``examples/`` (ask/tell Sphere, variant compare, literature smoke) with nbmake CI.
- Sphinx content for variants (primary citations), architecture layers, limitations, FAQ (ECabc disambiguation), and public API autodoc.
- Dependabot weekly updates for ``pip`` and ``github-actions``.

### Fixed

- qABC onlooker neighborhoods use per-source mean distance (Eq. 6) and apply candidates to the neighborhood-best index.
- Corrected Akay & Karaboga (2012) DOI to ``10.1016/j.ins.2010.07.015``.
