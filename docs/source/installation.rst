Installation
============

Requires Python 3.10 or newer.

From PyPI
---------

.. code-block:: bash

   pip install swarm-seek

Optional extras:

.. code-block:: bash

   pip install 'swarm-seek[numba]'    # Numba backend
   pip install 'swarm-seek[jax]'      # JAX backend
   pip install 'swarm-seek[sklearn]'  # ABCSearchCV
   pip install 'swarm-seek[optuna]'   # ABCSampler

From a clone
------------

.. code-block:: bash

   pip install -e ".[dev]"

Documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Backend selection
-----------------

Pass ``backend=`` to :class:`~swarm_seek.ABC` / :class:`~swarm_seek.colony.Colony`:

* ``"numpy"`` — reference implementation (always available).
* ``"numba"`` — requires ``swarm-seek[numba]``.
* ``"jax"`` — requires ``swarm-seek[jax]``; never chosen by ``"auto"``.
* ``"auto"`` — Numba if importable, otherwise NumPy.

A timing script for comparing backends lives at ``scripts/compare_backends.py``
(not a CI wall-clock gate).
