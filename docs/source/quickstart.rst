Quickstart
==========

Install the package (editable install shown for development):

.. code-block:: bash

   pip install -e ".[dev]"

Minimize the Sphere function with Original ABC:

.. code-block:: python

   from swarm_seek import ABC, ContinuousSpace
   from swarm_seek.benchmarks import sphere

   space = ContinuousSpace([(-5.0, 5.0)] * 5)
   colony = ABC(
       space,
       variant="original",
       pop_size=20,
       limit=100,
       max_evals=5_000,
       seed=0,
   )
   best = colony.minimize(sphere)
   print(best.fitness)

Ask/tell loop
-------------

.. code-block:: python

   import numpy as np

   from swarm_seek import ABC, ContinuousSpace
   from swarm_seek.benchmarks import sphere

   space = ContinuousSpace([(-5.0, 5.0)] * 5)
   colony = ABC(
       space,
       variant="gabc",
       pop_size=20,
       limit=100,
       max_evals=5_000,
       seed=0,
   )

   while not colony.converged:
       candidates = colony.ask()
       if candidates.size == 0:
           colony.tell([])
           continue
       colony.tell(np.asarray(sphere(candidates), dtype=np.float64))

   print(colony.best.fitness)

Examples
--------

Runnable notebooks with stated pass criteria are in the repository
``examples/`` directory (executed in CI with ``nbmake``):

* ``01_ask_tell_sphere.ipynb``
* ``02_variant_compare.ipynb``
* ``03_literature_smoke.ipynb``
* ``04_custom_objective.ipynb`` — wire your own fitness / ask/tell
* ``05_system_integration.ipynb`` — ``Colony`` orchestration for larger systems
