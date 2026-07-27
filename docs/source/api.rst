API reference
=============

Public root
-----------

Symbols re-exported from :mod:`swarm_seek` (see ``swarm_seek.__all__``).

.. automodule:: swarm_seek
   :members:
   :imported-members:
   :no-index:

Façade
------

.. autoclass:: swarm_seek.ABC
   :members:
   :show-inheritance:

Space
-----

.. autoclass:: swarm_seek.ContinuousSpace
   :members:
   :show-inheritance:

.. autoclass:: swarm_seek.PermutationSpace
   :members:
   :show-inheritance:

Integrations (optional extras)
------------------------------

.. autoclass:: swarm_seek.integrations.ABCSearchCV
   :members:
   :no-index:

.. autoclass:: swarm_seek.integrations.ABCSampler
   :members:
   :no-index:

Types
-----

.. autoclass:: swarm_seek.Solution
   :members:
   :show-inheritance:

Benchmarks
----------

Objectives and literature-oracle helpers used in validation notebooks and
tests.

.. automodule:: swarm_seek.benchmarks
   :members:
   :imported-members:

Power-user colony
-----------------

Most users should prefer :class:`~swarm_seek.ABC`. Direct construction of
:class:`~swarm_seek.colony.Colony` is supported for advanced budgeting and
introspection.

.. autoclass:: swarm_seek.colony.Colony
   :members:
   :show-inheritance:
