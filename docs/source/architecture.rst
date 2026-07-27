Architecture
============

Swarm Seek separates three orthogonal axes:

* **Space** — representation and bounds (``ContinuousSpace`` in ``0.1.0``).
* **Variant** — published ABC update equations (Original, GABC, qABC, MABC).
* **Backend** — array execution (NumPy default; Numba/JAX later).

Layer dependency rule
---------------------

Each layer imports only from the same layer or lower. Variants must not import
backends or the colony engine. Backends must not encode variant equations.
Cycles are forbidden.

.. code-block:: text

   L5  integrations/     (post-0.1.0: sklearn, optuna)
   L4  façade            ABC
   L3  colony            ask/tell engine
   L2  variants          strategy modules
   L1  backends          NumPy (default)
   L0  foundation        Space, Solution, RNG, errors
   L6  benchmarks        functions + literature oracles

Public vs power-user API
------------------------

* **Public façade** — symbols in ``swarm_seek.__all__``
  (``ABC``, ``ContinuousSpace``, ``Solution``, ``__version__``).
* **Documented subpackages** — e.g. ``swarm_seek.benchmarks``,
  ``swarm_seek.colony.Colony`` for advanced control.
* **Private helpers** — leading-underscore names; not part of the stable API.

Ask/tell contract
-----------------

:class:`~swarm_seek.ABC` (and :class:`~swarm_seek.colony.Colony`) alternate
``ask`` / ``tell`` through employed → onlooker → scout phases. ``ask`` returns
only candidates that need evaluation (design Q5). Convergence is controlled by
``max_evals``, ``max_iters``, and/or ``stall_evals``. Continuous bounds use
clip repair (design Q2).
