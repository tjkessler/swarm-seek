Installation
============

Requires Python 3.10 or newer.

From a clone
------------

.. code-block:: bash

   pip install -e ".[dev]"

Optional documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

From PyPI
---------

When published:

.. code-block:: bash

   pip install swarm-seek

Optional extras (``numba``, ``jax``, ``sklearn``, ``optuna``) are reserved for
post-``0.1.0`` backends and adapters.
