Limitations
===========

Current release (``0.1.1``)
----------------------------

* Continuous search spaces only (``ContinuousSpace``). Combinatorial ABC
  (CABC) is post-``0.1.0``.
* Single NumPy backend. Optional Numba/JAX backends are not shipped.
* No scikit-learn or Optuna adapters yet (designed as L5 extras later).
* Literature oracles use tolerance bands; papers typically omit RNG seeds, so
  bit-identical table reproduction is not claimed.

Non-goals
---------

Swarm Seek will **not**:

* Become a general metaheuristics suite (no PSO, DE, GA, … as first-class
  peers).
* Ship multi-objective / Pareto MOABC in ``0.1.0``.
* Require Numba, JAX, scikit-learn, or Optuna as hard dependencies.
* Replace domain-specific hyperparameter tuners; ask/tell is the integration
  surface for those tools.
