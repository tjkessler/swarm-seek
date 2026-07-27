Limitations
===========

Current release (``0.2.0``)
----------------------------

* Continuous backends: NumPy (default), optional Numba and JAX with numerical
  parity on ``generate_candidates``. ``backend="auto"`` prefers Numba when
  installed, otherwise NumPy; JAX is explicit opt-in only.
* Combinatorial search: ``PermutationSpace`` + ``variant="cabc"`` (swap /
  insertion neighborhoods per Karaboga & Gorkemli 2011). Binary/mixed spaces
  are not shipped.
* Adapters: ``ABCSearchCV`` (scikit-learn) and ``ABCSampler`` (Optuna) wrap
  ask/tell; categorical Optuna distributions are unsupported in ``0.2.0``.
* Literature oracles for continuous variants use tolerance bands; papers
  typically omit RNG seeds, so bit-identical table reproduction is not claimed.
  CABC ships a synthetic TSP smoke rather than a full paper-table oracle.

Non-goals
---------

Swarm Seek will **not**:

* Become a general metaheuristics suite (no PSO, DE, GA, … as first-class
  peers).
* Ship multi-objective / Pareto MOABC in the near term.
* Require Numba, JAX, scikit-learn, or Optuna as hard dependencies.
* Replace domain-specific hyperparameter tuners; ask/tell plus thin adapters
  are the integration surface for those tools.
