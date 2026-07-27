FAQ
===

How is Swarm Seek different from ECabc?
---------------------------------------

`ECabc <https://github.com/ECRL/ECabc>`_ is a JOSS-published package that uses
Artificial Bee Colony primarily as a **parameter / hyperparameter tuner**.

Swarm Seek is a general ABC **optimization library**: citation-linked variants
(Original, GABC, qABC, MABC), continuous search spaces, a NumPy ask/tell
engine, and literature-oracle regression tests.

Both may export a class named ``ABC``. Import paths differ:

* ``from swarm_seek import ABC``
* ``from ecabc import ABC``

Prefer the fully qualified names in prose when both packages appear in the same
project.

Why not use MEALPY or NiaPy?
----------------------------

Those frameworks ship the original ABC among many unrelated algorithms. Swarm
Seek focuses on a maintained, citation-linked ABC variant family with
literature-oracle CI and does not aim to be a general nature-inspired algorithm
suite.

Which variant should I start with?
----------------------------------

Use ``variant="original"`` for the classical Karaboga ABC baseline. Try
``gabc``, ``qabc``, or ``mabc`` when you need the corresponding published
search equation; see :doc:`variants` for citations and parameters.

How do literature oracles work?
-------------------------------

Fixtures under ``swarm_seek.benchmarks`` encode a paper table row (settings +
expected statistic + tolerance). CI runs a short multi-seed subset; full paper
run counts are available via ``@pytest.mark.slow``. Failures cite the fixture
id, DOI, and locator.
