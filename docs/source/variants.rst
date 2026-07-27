ABC variants
============

Swarm Seek ships continuous ABC strategies and combinatorial CABC as swappable
modules. Select a variant with ``ABC(..., variant="...")`` (or construct a
strategy via ``swarm_seek.variants.get_variant`` for power-user use).

.. list-table::
   :header-rows: 1
   :widths: 14 28 58

   * - Key
     - Name
     - Primary source
   * - ``original``
     - Original ABC
     - Karaboga, D. (2005). *An idea based on honey bee swarm for numerical
       function optimization* (Technical Report TR06). Erciyes University.
       Elaborated in Karaboga & Basturk (2007), *J. Glob. Optim.* 39:459–471.
   * - ``gabc``
     - Gbest-guided ABC
     - Zhu, G., & Kwong, S. (2010). Gbest-guided artificial bee colony
       algorithm for numerical function optimization. *Applied Mathematics and
       Computation*, 217(7), 3166–3173.
       https://doi.org/10.1016/j.amc.2010.08.049
   * - ``qabc``
     - Quick ABC
     - Karaboga, D., & Gorkemli, B. (2014). A quick artificial bee colony
       (qABC) algorithm and its performance on optimization problems.
       *Applied Soft Computing*, 23, 227–238.
       https://doi.org/10.1016/j.asoc.2014.06.035
   * - ``mabc``
     - Modified ABC
     - Akay, B., & Karaboga, D. (2012). A modified artificial bee colony
       algorithm for real-parameter optimization. *Information Sciences*,
       192, 120–142. https://doi.org/10.1016/j.ins.2010.07.015
   * - ``cabc``
     - Combinatorial ABC
     - Karaboga, D., & Gorkemli, B. (2011). A combinatorial Artificial Bee
       Colony algorithm for traveling salesman problem. *INISTA 2011*,
       pp. 50–53. https://doi.org/10.1109/INISTA.2011.5946125

Variant-specific parameters
---------------------------

Extra keyword arguments on :class:`~swarm_seek.ABC` are forwarded to the
strategy:

* **GABC** — ``C`` (default ``1.5``), upper bound on the gbest pull term.
* **qABC** — ``r`` (default ``1.0``), neighborhood radius factor for onlookers.
* **MABC** — ``MR`` (default ``0.4``) and ``SF`` (default ``1.0``), modification
  rate and scaling factor.
* **CABC** — ``operator`` (``"swap"``, ``"insertion"``, or ``"random"``;
  default ``"random"``). Use with :class:`~swarm_seek.PermutationSpace`.

Shared colony controls include ``pop_size``, abandonment ``limit``,
``max_evals`` / ``max_iters`` / ``stall_evals``, ``sense``, ``seed``, and
``backend`` (``"numpy"``, ``"numba"``, ``"jax"``, or ``"auto"``).

Literature oracles
------------------

Automated regression fixtures live under ``swarm_seek.benchmarks`` and assert
published table statistics within documented tolerance bands for the four
continuous variants. CABC validation uses a synthetic TSP tour-length smoke.
See ``examples/03_literature_smoke.ipynb`` for a short continuous smoke run.
