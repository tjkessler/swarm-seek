# Swarm Seek

Literature-validated Artificial Bee Colony (ABC) optimization for Python.

Swarm Seek implements Original ABC and principal peer-reviewed variants
(GABC, qABC, MABC, CABC) with a NumPy core, optional Numba/JAX backends, an
ask/tell API, scikit-learn / Optuna adapters, and automated checks against
published benchmark figures.

**Status:** `0.2.0` (beta). Docs: https://swarm-seek.readthedocs.io/

## Install

Requires Python 3.10+.

```bash
pip install swarm-seek
```

Optional extras:

```bash
pip install 'swarm-seek[numba]'    # Numba backend
pip install 'swarm-seek[jax]'      # JAX backend
pip install 'swarm-seek[sklearn]'  # ABCSearchCV
pip install 'swarm-seek[optuna]'   # ABCSampler
```

Editable install for development:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from swarm_seek import ABC, ContinuousSpace
from swarm_seek.benchmarks import sphere

space = ContinuousSpace([(-5.0, 5.0)] * 5)
colony = ABC(space, variant="original", pop_size=20, limit=100, max_evals=5_000, seed=0)
best = colony.minimize(sphere)
print(best.fitness)
```

Ask/tell loop:

```python
import numpy as np

from swarm_seek import ABC, ContinuousSpace
from swarm_seek.benchmarks import sphere

space = ContinuousSpace([(-5.0, 5.0)] * 5)
colony = ABC(space, variant="gabc", pop_size=20, limit=100, max_evals=5_000, seed=0)

while not colony.converged:
    candidates = colony.ask()
    if candidates.size == 0:
        colony.tell([])
        continue
    colony.tell(np.asarray(sphere(candidates), dtype=np.float64))

print(colony.best.fitness)
```

Combinatorial ABC (permutation / TSP-style):

```python
from swarm_seek import ABC, PermutationSpace
from swarm_seek.benchmarks import random_cities, tour_length

cities = random_cities(12, seed=0)
space = PermutationSpace(12)
best = ABC(space, variant="cabc", pop_size=16, max_evals=2_000, seed=0).minimize(
    lambda x: tour_length(x, cities)
)
print(best.fitness)
```

Runnable notebooks with stated pass criteria live in [`examples/`](examples/).
Backend timing: `python scripts/compare_backends.py`.
Full documentation builds from `docs/source/`.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata. Archive / DOI:
[10.5281/zenodo.21630532](https://doi.org/10.5281/zenodo.21630532)
(concept; version v0.2.0 is
[10.5281/zenodo.21631303](https://doi.org/10.5281/zenodo.21631303)).
