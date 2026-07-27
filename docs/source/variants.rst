ABC variants
============

Swarm Seek ships four continuous ABC strategies as swappable modules.
Select a variant with ``ABC(..., variant="...")`` (or construct a strategy
via ``swarm_seek.variants.get_variant`` for power-user use).

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

Variant-specific parameters
---------------------------

Extra keyword arguments on :class:`~swarm_seek.ABC` are forwarded to the
strategy:

* **GABC** — ``C`` (default ``1.5``), upper bound on the gbest pull term.
* **qABC** — ``r`` (default ``1.0``), neighborhood radius factor for onlookers.
* **MABC** — ``MR`` (default ``0.4``) and ``SF`` (default ``1.0``), modification
  rate and scaling factor.

Shared colony controls include ``pop_size``, abandonment ``limit``,
``max_evals`` / ``max_iters`` / ``stall_evals``, ``sense``, and ``seed``.

Literature oracles
------------------

Automated regression fixtures live under ``swarm_seek.benchmarks`` and assert
published table statistics within documented tolerance bands. See the
``examples/03_literature_smoke.ipynb`` notebook for a short smoke run.
Combinatorial ABC (CABC) and other post-``0.1.0`` variants are out of scope
for this release.
